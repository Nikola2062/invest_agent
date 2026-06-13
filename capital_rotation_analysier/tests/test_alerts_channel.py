"""Tests for the Telegram channel hardening (truncation + Markdown fallback)
and the §1.7 coverage gate in the shared ingest runner."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import pytest

from rotation.alerts import (
    Alert, TELEGRAM_MAX_TEXT, _telegram_send, _truncate_for_telegram,
)
from rotation.config import (
    Config, IngestConfig, RetryPolicy, StorageConfig, Symbol,
)
from rotation.ingest.runner import fetch_validate_store
from rotation.store import connect


# ------------------------------------------------------------
# Telegram truncation
# ------------------------------------------------------------

def test_truncate_noop_below_limit():
    assert _truncate_for_telegram("short") == "short"


def test_truncate_caps_at_telegram_limit():
    long = "x" * (TELEGRAM_MAX_TEXT + 500)
    out = _truncate_for_telegram(long)
    assert len(out) <= TELEGRAM_MAX_TEXT
    assert out.endswith("… (truncated)")


def test_telegram_send_truncates_and_falls_back_to_plain_text(monkeypatch):
    """First POST (Markdown) is rejected; the retry must drop parse_mode and
    the oversized body must be truncated below the API limit."""
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")

    posts: list[dict] = []

    class _Resp:
        status = 200
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def fake_urlopen(req, timeout=None):
        import urllib.parse
        payload = dict(urllib.parse.parse_qsl(req.data.decode()))
        posts.append(payload)
        if "parse_mode" in payload:
            raise OSError("HTTP 400: can't parse entities")
        return _Resp()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    alert = Alert(
        alert_type="validation_failure", priority="P1",
        headline="signal failed: risk_on_off",  # underscores break Markdown
        body="detail\n" * 2000,                 # far beyond 4096 chars
        ts=date(2026, 6, 10),
    )
    out = _telegram_send(alert)

    assert out == "telegram:200"
    assert len(posts) == 2, "must retry exactly once without parse_mode"
    assert "parse_mode" in posts[0] and "parse_mode" not in posts[1]
    assert all(len(p["text"]) <= TELEGRAM_MAX_TEXT for p in posts)


def test_telegram_send_skips_without_creds(monkeypatch):
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    a = Alert("t", "P3", "h", "b", date(2026, 6, 10))
    assert _telegram_send(a) is None


# ------------------------------------------------------------
# Coverage gate (design §1.7)
# ------------------------------------------------------------

def _cfg(tmp_path: Path, min_coverage_pct: float = 90.0) -> Config:
    return Config(
        storage=StorageConfig(duckdb_path=tmp_path / "rot.duckdb"),
        ingest=IngestConfig(
            primary_source="yfinance",
            retry=RetryPolicy(max_attempts=1, backoff_seconds=(1,)),
            stale_threshold_days=4,
            outlier_intraday_pct=20.0,
            outlier_intraday_pct_by_class={},
            min_coverage_pct=min_coverage_pct,
        ),
        universe=(
            Symbol("SPY", "equity_us"),
            Symbol("QQQ", "equity_us"),
            Symbol("TLT", "bond"),
            Symbol("GLD", "commodity_precious"),
            Symbol("0700.HK", "equity_hk"),   # exempt from the gate
        ),
    )


def _bar(symbol: str, asof: date) -> dict:
    return {
        "symbol": symbol, "asset_class": "equity_us", "ts": asof,
        "open": 99.0, "high": 101.0, "low": 98.5, "close": 100.0,
        "adj_close": 100.0, "volume": 1_000_000.0,
        "source": "test", "ingested_at": datetime(2026, 6, 9, 22, 0),
        "stale": False,
    }


def test_full_coverage_is_ok(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    asof = date(2026, 6, 9)
    bars = [_bar(s, asof) for s in ("SPY", "QQQ", "TLT", "GLD")]
    monkeypatch.setattr("rotation.ingest.runner.fetch_bars", lambda c, a: bars)

    with connect(cfg.storage.duckdb_path) as con:
        out = fetch_validate_store(cfg, con, "test-run-1", asof)
        status = con.execute(
            "SELECT status FROM run_log WHERE run_id = 'test-run-1'"
        ).fetchone()[0]

    assert out["coverage_ok"] is True
    assert out["coverage"] == 1.0
    assert status == "ok"


def test_thin_coverage_marks_run_degraded(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    asof = date(2026, 6, 9)
    bars = [_bar("SPY", asof)]  # 1 of 4 gated symbols
    monkeypatch.setattr("rotation.ingest.runner.fetch_bars", lambda c, a: bars)

    with connect(cfg.storage.duckdb_path) as con:
        out = fetch_validate_store(cfg, con, "test-run-2", asof)
        status, notes = con.execute(
            "SELECT status, notes FROM run_log WHERE run_id = 'test-run-2'"
        ).fetchone()

    assert out["coverage_ok"] is False
    assert out["coverage"] == 0.25
    assert status == "degraded"
    assert "coverage" in notes


def test_hk_symbols_do_not_count_toward_gate(tmp_path, monkeypatch):
    """An HKEX holiday (no HK bars) must not degrade a clean NYSE run."""
    cfg = _cfg(tmp_path)
    asof = date(2026, 6, 9)
    bars = [_bar(s, asof) for s in ("SPY", "QQQ", "TLT", "GLD")]  # no 0700.HK
    monkeypatch.setattr("rotation.ingest.runner.fetch_bars", lambda c, a: bars)

    with connect(cfg.storage.duckdb_path) as con:
        out = fetch_validate_store(cfg, con, "test-run-3", asof)

    assert out["coverage_ok"] is True
    assert out["coverage"] == 1.0


def test_lookback_bars_do_not_mask_missing_asof_bars(tmp_path, monkeypatch):
    """Provider returns the lookback days fine but nothing for asof itself
    (observed live: Yahoo serving NaN closes for the latest session). The gate
    must measure coverage AT asof, not anywhere in the fetch window."""
    cfg = _cfg(tmp_path)
    asof = date(2026, 6, 9)
    prior = date(2026, 6, 8)
    bars = [_bar(s, prior) for s in ("SPY", "QQQ", "TLT", "GLD")]  # no asof rows
    monkeypatch.setattr("rotation.ingest.runner.fetch_bars", lambda c, a: bars)

    with connect(cfg.storage.duckdb_path) as con:
        out = fetch_validate_store(cfg, con, "test-run-5", asof)

    assert out["coverage"] == 0.0
    assert out["coverage_ok"] is False


def test_fetch_error_logs_failed_run(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    asof = date(2026, 6, 9)

    def boom(c, a):
        raise RuntimeError("yfinance down")
    monkeypatch.setattr("rotation.ingest.runner.fetch_bars", boom)

    with connect(cfg.storage.duckdb_path) as con:
        with pytest.raises(RuntimeError):
            fetch_validate_store(cfg, con, "test-run-4", asof)
        status = con.execute(
            "SELECT status FROM run_log WHERE run_id = 'test-run-4'"
        ).fetchone()[0]

    assert status == "failed"
