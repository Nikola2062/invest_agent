"""EDGAR provider tests — all HTTP mocked.

The headline claim Phase 4+ rests on is **filing-date PIT correctness**:
a Dec 31 2022 10-K does NOT appear in a 2023-01-15 query because it
wasn't filed until late February 2023. The tests below pin that
guarantee plus the surrounding plumbing (CIK lookup, rate limiter,
User-Agent, restatement handling).
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from tradingagents.dataflows import edgar
from tradingagents.dataflows.edgar import (
    StatementPoint,
    _RateLimiter,
    _filter_by_filing_date,
    _latest_per_period,
    _user_agent,
)

pytestmark = pytest.mark.unit


# --- helpers --------------------------------------------------------------


def _facts_with_concepts(concept_entries: dict[str, list[dict]]) -> dict:
    """Build a fake companyfacts blob containing the given concept entries.

    Each entry is the raw shape EDGAR returns:
      {"end": "...", "filed": "...", "form": "...", "val": ...}
    """
    units = {
        concept: {"units": {"USD": entries}}
        for concept, entries in concept_entries.items()
    }
    return {"facts": {"us-gaap": units}}


@pytest.fixture(autouse=True)
def _isolated_cache(tmp_path, monkeypatch):
    """Each test gets its own data_cache_dir + cleared CIK cache."""
    monkeypatch.setattr(edgar, "_CIK_CACHE", None)
    config_patch = {"data_cache_dir": str(tmp_path)}
    monkeypatch.setattr(edgar, "get_config", lambda: config_patch)
    yield


# --- rate limiter --------------------------------------------------------


def test_rate_limiter_serialises_calls():
    """Two acquires must be spaced at least 1/rps apart."""
    limiter = _RateLimiter(rps=20.0)  # 50ms per call
    t0 = time.monotonic()
    limiter.acquire()
    limiter.acquire()
    elapsed = time.monotonic() - t0
    assert elapsed >= 0.045  # ≥ 50ms (with a touch of slack for clock noise)


def test_rate_limiter_is_thread_safe():
    """N threads each doing one acquire should still serialise correctly."""
    limiter = _RateLimiter(rps=50.0)
    n = 6
    deltas = []
    last = [time.monotonic()]
    lock = threading.Lock()

    def worker():
        limiter.acquire()
        with lock:
            now = time.monotonic()
            deltas.append(now - last[0])
            last[0] = now

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # n-1 inter-call gaps should each be ≥ ~20ms (1/50).
    inter_call_gaps = deltas[1:]
    assert all(d >= 0.015 for d in inter_call_gaps)


# --- User-Agent -----------------------------------------------------------


def test_user_agent_falls_back_with_warning(caplog, monkeypatch):
    monkeypatch.delenv("EDGAR_USER_AGENT", raising=False)
    monkeypatch.setattr(edgar, "get_config", lambda: {"edgar_user_agent": ""})
    with caplog.at_level("WARNING"):
        ua = _user_agent()
    assert ua  # non-empty fallback
    assert any("EDGAR_USER_AGENT not set" in r.message for r in caplog.records)


def test_user_agent_from_env_overrides_config(monkeypatch):
    monkeypatch.setenv("EDGAR_USER_AGENT", "env-name env@example.com")
    monkeypatch.setattr(edgar, "get_config", lambda: {"edgar_user_agent": "config-only"})
    assert _user_agent() == "env-name env@example.com"


# --- CIK lookup -----------------------------------------------------------


def test_cik_lookup_zero_pads_and_caches(tmp_path, monkeypatch):
    monkeypatch.setattr(edgar, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    fake_index = {
        "0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."},
        "1": {"cik_str": 789019, "ticker": "MSFT", "title": "Microsoft Corp"},
    }

    call_count = {"n": 0}

    def fake_get_json(url):
        call_count["n"] += 1
        return fake_index

    with patch.object(edgar, "_get_json", side_effect=fake_get_json):
        assert edgar.lookup_cik("AAPL") == "0000320193"
        assert edgar.lookup_cik("MSFT") == "0000789019"
        # In-memory cache means a second lookup doesn't refetch.
        assert edgar.lookup_cik("AAPL") == "0000320193"
        assert call_count["n"] == 1


def test_cik_lookup_unknown_ticker_returns_none(tmp_path, monkeypatch):
    monkeypatch.setattr(edgar, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    with patch.object(edgar, "_get_json", return_value={}):
        assert edgar.lookup_cik("NOSUCHTICKER") is None
        assert edgar.lookup_cik("") is None


def test_cik_cache_persists_across_processes(tmp_path, monkeypatch):
    monkeypatch.setattr(edgar, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    fake_index = {"0": {"cik_str": 1, "ticker": "AAA", "title": "A"}}
    with patch.object(edgar, "_get_json", return_value=fake_index):
        edgar.lookup_cik("AAA")

    # Clear in-memory cache to simulate a fresh process.
    edgar._CIK_CACHE = None
    # Next lookup must NOT call _get_json — it should hit the disk cache.
    with patch.object(edgar, "_get_json", side_effect=AssertionError("must not refetch")):
        assert edgar.lookup_cik("AAA") == "0000000001"


# --- filing-date PIT filter — the headline claim --------------------------


def _pt(period_end, filed, val=100.0, concept="Revenues", form="10-K", label="Revenue"):
    return StatementPoint(
        concept=concept, label=label, value=val,
        period_end=period_end, filed=filed, form=form, unit="USD",
    )


def test_filter_drops_filings_after_curr_date():
    pts = [
        _pt("2022-12-31", "2023-02-15"),
        _pt("2023-03-31", "2023-05-01"),
        _pt("2023-06-30", "2023-08-01"),
    ]
    out = _filter_by_filing_date(pts, "2023-04-15")
    assert [p.period_end for p in out] == ["2022-12-31"]


def test_filter_hides_unfiled_annual():
    """A 2022 10-K with period_end=2022-12-31 but filed=2023-02-28 must
    NOT appear in a query on 2023-01-15."""
    pts = [_pt("2022-12-31", "2023-02-28")]
    out = _filter_by_filing_date(pts, "2023-01-15")
    assert out == []


def test_filter_with_empty_curr_date_is_passthrough():
    pts = [_pt("2022-12-31", "2023-02-28"), _pt("2023-03-31", "2023-05-01")]
    assert len(_filter_by_filing_date(pts, "")) == 2
    assert len(_filter_by_filing_date(pts, None)) == 2


def test_latest_per_period_takes_restated_value():
    """When an amended 10-K (10-K/A) follows the original with a different
    value, the amended (later-filed) one wins."""
    pts = [
        _pt("2022-12-31", "2023-02-15", val=100.0, form="10-K"),
        _pt("2022-12-31", "2023-08-01", val=120.0, form="10-K/A"),  # restatement
    ]
    out = _latest_per_period(pts)
    assert len(out) == 1
    assert out[0].value == 120.0
    assert out[0].form == "10-K/A"


# --- end-to-end statement extraction --------------------------------------


def test_get_balance_sheet_respects_pit_filter(tmp_path, monkeypatch):
    """Filed-date filtering survives the full get_balance_sheet pipeline."""
    monkeypatch.setattr(edgar, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    # Two fiscal periods. The newer one's 10-K hasn't been filed yet by curr_date.
    facts = _facts_with_concepts({
        "Assets": [
            {"end": "2022-12-31", "filed": "2023-02-28", "form": "10-K", "val": 1000},
            {"end": "2023-12-31", "filed": "2024-02-28", "form": "10-K", "val": 1200},
        ],
        "Liabilities": [
            {"end": "2022-12-31", "filed": "2023-02-28", "form": "10-K", "val": 500},
            {"end": "2023-12-31", "filed": "2024-02-28", "form": "10-K", "val": 600},
        ],
    })

    with patch.object(edgar, "fetch_companyfacts", return_value=facts):
        out = edgar.get_balance_sheet("AAPL", freq="annual", curr_date="2024-01-15")

    assert "2022-12-31" in out
    # The 2023 fiscal-year statement was filed 2024-02-28 — must not appear
    # on a 2024-01-15 query.
    assert "2023-12-31" not in out
    assert "Source: SEC EDGAR" in out


def test_get_balance_sheet_passes_through_when_curr_date_in_future(tmp_path, monkeypatch):
    monkeypatch.setattr(edgar, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    facts = _facts_with_concepts({
        "Assets": [
            {"end": "2022-12-31", "filed": "2023-02-28", "form": "10-K", "val": 1000},
        ],
    })
    with patch.object(edgar, "fetch_companyfacts", return_value=facts):
        out = edgar.get_balance_sheet("AAPL", freq="annual", curr_date="2099-01-01")
    assert "2022-12-31" in out


def test_get_income_statement_uses_quarterly_forms(tmp_path, monkeypatch):
    """freq=quarterly must select 10-Q forms only."""
    monkeypatch.setattr(edgar, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    facts = _facts_with_concepts({
        "Revenues": [
            {"end": "2023-03-31", "filed": "2023-05-01", "form": "10-Q", "val": 100},
            {"end": "2022-12-31", "filed": "2023-02-28", "form": "10-K", "val": 400},
        ],
    })
    with patch.object(edgar, "fetch_companyfacts", return_value=facts):
        out = edgar.get_income_statement("AAPL", freq="quarterly", curr_date="2024-01-01")
    assert "2023-03-31" in out
    # The 10-K is annual; must not appear in a quarterly statement.
    assert "2022-12-31" not in out


def test_get_fundamentals_renders_pit_safe_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(edgar, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    facts = _facts_with_concepts({
        "Revenues": [
            {"end": "2023-12-31", "filed": "2024-02-28", "form": "10-K", "val": 1000},
            {"end": "2022-12-31", "filed": "2023-02-28", "form": "10-K", "val": 800},
        ],
        "NetIncomeLoss": [
            {"end": "2023-12-31", "filed": "2024-02-28", "form": "10-K", "val": 200},
            {"end": "2022-12-31", "filed": "2023-02-28", "form": "10-K", "val": 150},
        ],
    })
    with patch.object(edgar, "fetch_companyfacts", return_value=facts):
        # On 2024-01-15: only the 2022 10-K is public. The 2023 10-K wasn't
        # filed until 2024-02-28.
        out = edgar.get_fundamentals("AAPL", curr_date="2024-01-15")
    assert "period 2022-12-31" in out
    assert "filed 2023-02-28" in out
    assert "period 2023-12-31" not in out
    assert "EDGAR Fundamentals for AAPL" in out


def test_get_fundamentals_handles_unknown_ticker(tmp_path, monkeypatch):
    monkeypatch.setattr(edgar, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    with patch.object(edgar, "fetch_companyfacts", return_value=None):
        out = edgar.get_fundamentals("NOSUCHTICKER", curr_date="2024-01-15")
    assert "No EDGAR fundamentals" in out


def test_get_balance_sheet_handles_empty_facts(tmp_path, monkeypatch):
    """A company with no us-gaap facts must return a clean message, not crash."""
    monkeypatch.setattr(edgar, "get_config", lambda: {"data_cache_dir": str(tmp_path)})
    with patch.object(edgar, "fetch_companyfacts", return_value={"facts": {}}):
        out = edgar.get_balance_sheet("EMPTY", freq="annual", curr_date="2024-01-15")
    assert "No EDGAR" in out


# --- PIT provider marker -------------------------------------------------


def test_edgar_marks_fundamentals_pit_safe():
    """The provider registry must continue to mark fundamentals PIT-safe.
    If this changes, audit before relaxing the guarantee."""
    from tradingagents.dataflows.providers import pit_safe_methods
    safe = pit_safe_methods("edgar")
    assert "get_fundamentals" in safe
    assert "get_balance_sheet" in safe
    assert "get_income_statement" in safe
    assert "get_cashflow" in safe
