"""Tests for the wishlist W4/W5/W6 batch:

  W5 — Executive Dashboard (cover-page decision table, US + HK)
  W4 — Signal Inflection Monitor (5d-ago vs now, US + HK)
  W6 — HK Relative Rotation Forecast (Leadership Persistence Score)
"""
from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from rotation.config import (
    Config, IngestConfig, RetryPolicy, StorageConfig, Symbol,
)
from rotation.report import (
    _flow_direction,
    _recommended_exposure,
    _risk_posture,
    _trajectory,
    _trend_arrow,
    section_executive_dashboard,
    section_inflection_monitor,
)
from rotation.store import connect


def _cfg(tmp_path) -> Config:
    return Config(
        storage=StorageConfig(duckdb_path=tmp_path / "rot.duckdb"),
        ingest=IngestConfig(
            primary_source="yfinance",
            retry=RetryPolicy(max_attempts=1, backoff_seconds=(1,)),
            stale_threshold_days=4,
            outlier_intraday_pct=20.0,
            outlier_intraday_pct_by_class={},
        ),
        universe=(Symbol("SPY", "equity_index"),),
    )


def _metrics() -> pd.DataFrame:
    return pd.DataFrame({
        "symbol": ["XLV", "XLP", "SMH", "XLK"],
        "r_d": [0.002, 0.001, -0.004, -0.003],
        "r_w": [0.012, 0.008, -0.015, -0.011],
        "r_m": [0.03, 0.02, -0.04, -0.03],
        "rs_accel_5": [18.0, 4.0, -22.0, -9.0],
    })


# ============================================================
# W5 helper math
# ============================================================

def test_risk_posture_and_trajectory():
    now = {"risk_on_off": {"score": -10.0}, "capital_rotation": {"score": -20.0},
           "growth": {"score": -30.0}, "recession": {"score": 30.0}}
    prev = {"risk_on_off": {"score": 20.0}, "capital_rotation": {"score": 20.0},
            "growth": {"score": 20.0}, "recession": {"score": 0.0}}
    # now posture = mean(-10,-20,-30) - 30 = -20 - 30 = -50
    assert _risk_posture(now) == -50.0
    # prev posture = 20 - 0 = 20
    assert _risk_posture(prev) == 20.0
    label, d = _trajectory(now, prev)
    assert "Deteriorating" in label and d == -70.0


def test_trajectory_no_history():
    assert _trajectory({"growth": {"score": 5.0}}, None) == ("—", None)


def test_flow_direction():
    assert _flow_direction({"capital_rotation": {"score": 32.0}}, pd.DataFrame()) == "Risk-On Rotation"
    assert _flow_direction({"capital_rotation": {"score": -40.0}}, pd.DataFrame()) == "Defensive Rotation"
    # Near-zero rotation, defensive bucket leading -> defensive bid
    buckets = pd.DataFrame(
        {"r_w_mean": [0.012], "risk_tag": ["defensive"]}, index=["Healthcare"]
    )
    assert _flow_direction({"capital_rotation": {"score": 2.0}}, buckets) == "Defensive bid"


def test_recommended_exposure_reduce_and_add():
    sig_bad = {"recession": {"score": 25.0}, "liquidity": {"score": 35.0}}
    assert _recommended_exposure(sig_bad, "Deteriorating ↓", {}) == "Reduce Risk ↓"
    sig_good = {"recession": {"score": -25.0}, "liquidity": {"score": 65.0}}
    assert _recommended_exposure(sig_good, "Improving ↑",
                                 {"bullish_pct": 0.7, "bearish_pct": 0.1}) == "Add Risk ↑"
    assert _recommended_exposure({}, "Stable →", {}) == "Maintain →"


def test_trend_arrow():
    assert _trend_arrow(None) == "—"
    assert _trend_arrow(25.0) == "⇈"
    assert _trend_arrow(5.0) == "↑"
    assert _trend_arrow(-25.0) == "⇊"
    assert _trend_arrow(-5.0) == "↓"
    assert _trend_arrow(0.0) == "→"


# ============================================================
# W5 — Executive Dashboard section
# ============================================================

def test_executive_dashboard_renders(tmp_path):
    cfg = _cfg(tmp_path)
    with connect(cfg.storage.duckdb_path):
        pass  # schema only — no analogue history
    signals = {
        "capital_rotation": {"score": -32.0, "confidence": 0.6},
        "risk_on_off": {"score": -20.0, "confidence": 0.6},
        "recession": {"score": 25.0, "confidence": 0.5},
        "liquidity": {"score": 35.0, "confidence": 0.5},
    }
    prev_w = {"capital_rotation": {"score": 10.0}, "risk_on_off": {"score": 10.0},
              "recession": {"score": 0.0}}
    regime = {"regime": "Risk-Off Contraction", "days_in_regime": 3}
    md = section_executive_dashboard(cfg, date(2026, 6, 9), signals, prev_w,
                                     _metrics(), regime, None)
    assert "## Executive Dashboard" in md
    assert "Current Market State | Risk-Off Contraction" in md
    assert "Deteriorating" in md
    assert "Defensive Rotation" in md
    assert "Strongest Sector | Healthcare" in md
    assert "Weakest Sector | Technology" in md
    assert "Recommended Risk Exposure | Reduce Risk" in md
    assert "insufficient analogue history" in md  # no analogue store
    assert "not investment advice" in md.lower()


def test_executive_dashboard_empty_inputs(tmp_path):
    cfg = _cfg(tmp_path)
    with connect(cfg.storage.duckdb_path):
        pass
    md = section_executive_dashboard(cfg, date(2026, 6, 9), {}, None,
                                     pd.DataFrame(), None, None)
    assert "## Executive Dashboard" in md
    assert "Current Market State | —" in md
    assert "Signal Confidence | —" in md


# ============================================================
# W4 — Signal Inflection Monitor section
# ============================================================

def test_inflection_monitor_renders_arrows():
    signals = {"growth": {"score": -46.0}, "inflation": {"score": -58.0},
               "recession": {"score": 40.0}, "liquidity": {"score": 38.0}}
    prev_w = {"growth": {"score": 70.0}, "inflation": {"score": -20.0},
              "recession": {"score": 15.0}, "liquidity": {"score": 42.0}}
    md = section_inflection_monitor(signals, prev_w, _metrics())
    assert "## Signal Inflection Monitor" in md
    assert "| Growth | +70.0 | -46.0 | -116.0 | ⇊ |" in md
    assert "| Recession | +15.0 | +40.0 | +25.0 | ⇈ |" in md
    # Per-asset Δ²RS companion
    assert "Fastest accelerating" in md and "XLV" in md
    assert "Fastest decelerating" in md and "SMH" in md


def test_inflection_monitor_no_history():
    md = section_inflection_monitor({"growth": {"score": 1.0}}, None, pd.DataFrame())
    assert "needs a week of signal history" in md


# ============================================================
# W6 — HK Relative Rotation Forecast (Leadership Persistence)
# ============================================================

from rotation.report_hk import (  # noqa: E402
    _leadership_persistence,
    _section_hk_executive_dashboard,
    _section_hk_inflection_monitor,
    _section_hk_rotation_forecast,
    _section_hk_sector_forecast,
)


def _hk_metrics() -> pd.DataFrame:
    # 6 HK names spanning leaders (high RS, rising, accelerating) to laggards.
    return pd.DataFrame({
        "symbol": ["1398.HK", "0939.HK", "0700.HK", "9988.HK", "9888.HK", "1810.HK"],
        "close": [5.0, 5.5, 380.0, 78.0, 70.0, 12.0],
        "r_d": [0.01, 0.008, 0.004, -0.006, -0.008, -0.01],
        "r_w": [0.03, 0.02, 0.01, -0.02, -0.03, -0.04],
        "r_m": [0.06, 0.05, 0.02, -0.03, -0.05, -0.06],
        "rs_rank": [92.0, 85.0, 70.0, 30.0, 20.0, 12.0],
        "rs_change_1": [3.0, 2.0, 1.0, -1.0, -2.0, -3.0],
        "rs_change_5": [20.0, 15.0, 5.0, -10.0, -18.0, -25.0],
        "rs_change_21": [30.0, 22.0, 8.0, -12.0, -20.0, -28.0],
        "rs_accel_5": [12.0, 8.0, 2.0, -6.0, -10.0, -15.0],
    })


def test_leadership_persistence_orders_leaders_first():
    ranked = _leadership_persistence(_hk_metrics())
    # Strong-and-accelerating name ranks first; weakest ranks last.
    assert ranked.iloc[0]["symbol"] == "1398.HK"
    assert ranked.iloc[-1]["symbol"] == "1810.HK"
    # Monotonic non-increasing persistence.
    p = ranked["persistence"].tolist()
    assert p == sorted(p, reverse=True)


def test_leadership_persistence_drops_short_history():
    m = _hk_metrics()
    m.loc[0, "rs_accel_5"] = None  # one name missing an input -> dropped
    ranked = _leadership_persistence(m)
    assert "1398.HK" not in set(ranked["symbol"])


def test_hk_rotation_forecast_section():
    md = _section_hk_rotation_forecast(_hk_metrics(), n=11)
    assert "## Section 11 — Where Money Likely Goes Next" in md
    assert "Heuristic" in md
    assert "### Likely Leaders (Next 20 Sessions)" in md
    assert "### Likely Laggards (Next 20 Sessions)" in md
    assert "1398.HK" in md and "1810.HK" in md
    assert "no** forward-return validation" in md.lower() or "no forward-return" in md.lower()


def test_hk_rotation_forecast_insufficient_history():
    md = _section_hk_rotation_forecast(pd.DataFrame(), n=11)
    assert "Insufficient RS history" in md


def test_hk_sector_forecast_section():
    md = _section_hk_sector_forecast(_hk_metrics(), n=13)
    assert "## Section 13 — Sector Forecast" in md
    assert "Mean Persistence" in md
    assert "Lead 1" in md


def test_hk_executive_dashboard():
    md = _section_hk_executive_dashboard(_hk_metrics(), pd.DataFrame(), date(2026, 6, 9))
    assert "## Executive Dashboard" in md
    assert "Current Market State" in md
    assert "Likely Leader (20D, heuristic) | 1398.HK" in md
    assert "Likely Laggard (20D, heuristic) | 1810.HK" in md
    assert "HKEX Session | 2026-06-09" in md


def test_hk_inflection_monitor():
    md = _section_hk_inflection_monitor(_hk_metrics())
    assert "## Signal Inflection Monitor" in md
    assert "Net 5d RS momentum" in md
    assert "⇈ accel" in md and "⇊ decel" in md
    assert "1398.HK" in md


# ============================================================
# W8 — macro context narrative (FRED)
# ============================================================

from rotation.report import _macro_context_lines, section_explanations  # noqa: E402


def _fred_panel() -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=80, freq="B")
    df = pd.DataFrame(index=idx)
    df["DFEDTARU"] = 5.50  # held flat over the window
    df.iloc[:20, df.columns.get_loc("DFEDTARU")] = 5.75  # was higher a quarter ago
    df["T10Y2Y"] = -0.45   # inverted curve
    df["BAMLH0A0HYM2"] = 3.0
    df.iloc[-1, df.columns.get_loc("BAMLH0A0HYM2")] = 3.4  # widening 40bps over window
    df["T10YIE"] = 2.30
    return df


def test_macro_context_lines():
    lines = _macro_context_lines(_fred_panel())
    text = " ".join(lines)
    assert "Fed funds target (upper bound) **5.50%**" in text
    assert "-25 bps over ~1 quarter" in text
    assert "inverted" in text and "-45 bps" in text
    assert "HY credit spread (OAS) **3.40%**" in text
    assert "10-year breakeven inflation **2.30%**" in text


def test_macro_context_empty():
    assert _macro_context_lines(None) == []
    assert _macro_context_lines(pd.DataFrame()) == []


def test_section_explanations_appends_macro():
    interp = {"source": "rules-based", "claims": []}
    md = section_explanations(interp, None, n=25, fred=_fred_panel())
    assert "Macro context (observable, FRED)" in md
    # No fred -> no macro block (back-compat with the positional call).
    md2 = section_explanations(interp, None)
    assert "Macro context" not in md2


# ============================================================
# C2 — per-section resilience in the report builders
# ============================================================

from rotation.report import _safe_section  # noqa: E402


def test_safe_section_isolates_failure():
    ok = _safe_section("Section 9 — Whatever", lambda: "\n## Section 9 — Whatever\n\nbody")
    assert "body" in ok

    def boom():
        raise ValueError("kaboom")

    placeholder = _safe_section("Section 9 — Whatever", boom)
    assert "## Section 9 — Whatever — unavailable" in placeholder
    assert "ValueError: kaboom" in placeholder


def test_report_survives_a_broken_section(tmp_path, monkeypatch):
    """A section raising must not abort the whole report (C2)."""
    import rotation.report as R
    cfg = _cfg(tmp_path)
    with connect(cfg.storage.duckdb_path):
        pass  # empty schema — most sections render their 'no data' branch

    monkeypatch.setattr(
        R, "section_market_regime",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("regime exploded")))
    body = R.build_daily_report(cfg, date(2026, 6, 9))
    # The broken section degrades to a placeholder…
    assert "Section 3 — Market Regime — unavailable" in body
    # …but the rest of the report (and the glossary) still renders.
    assert "## Section 1 — Overview" in body
    assert "## Section 27 — Glossary" in body


# ============================================================
# C3 — a failure after log_run_start must not orphan the run_log row
# ============================================================

def test_run_log_not_orphaned_on_post_fetch_failure(tmp_path, monkeypatch):
    import rotation.ingest.runner as RUN
    cfg = _cfg(tmp_path)
    # Fetch succeeds (returns nothing); the failure lands in upsert — i.e. AFTER
    # the point the old code's narrow try guarded.
    monkeypatch.setattr(RUN, "fetch_bars", lambda cfg, asof: [])
    monkeypatch.setattr(RUN, "upsert_bars",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    with connect(cfg.storage.duckdb_path) as con:
        with pytest.raises(RuntimeError):
            RUN.fetch_validate_store(cfg, con, "run-c3", date(2026, 6, 9))
        status = con.execute(
            "SELECT status FROM run_log WHERE run_id = 'run-c3'").fetchone()[0]
    assert status == "failed"  # not the orphaned 'running'
