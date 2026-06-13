"""Tests for the wishlist W1/W2/W3 batch:

  W3 — per-driver signal attribution (signals.py + report section)
  W1 — forecast scorecard (scorecard.py + forecasts table + report section)
  W2 — Investment Committee View (US + HK report sections)
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from rotation import scorecard as SC
from rotation import signals as S
from rotation.config import (
    Config, IngestConfig, RetryPolicy, StorageConfig, Symbol,
)
from rotation.report import (
    section_committee_view,
    section_forecast_scorecard,
    section_signal_attribution,
)
from rotation.store import connect, upsert_bars


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


# ============================================================
# W3 — attribution math in signals.py
# ============================================================

def _panel(n_days: int = 320, seed: int = 3):
    """Synthetic close/volume panel covering every composite's inputs."""
    syms = sorted(set(
        S.RISK_ON_BASKET + S.RISK_OFF_BASKET + S.GROWTH_LEADERS + S.GROWTH_LAGGARDS
        + ["USO", "SLV", "LQD", "XLY", "XLP", "XLU", "SMH", "QQQ", "SPY", "UUP",
           "KWEB", "FXI", "MCHI", "EZU", "EWJ", "IEF"]
    ))
    idx = pd.date_range("2025-01-02", periods=n_days, freq="B")
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0004, 0.012, size=(n_days, len(syms)))
    close = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=idx, columns=syms)
    volume = pd.DataFrame(
        rng.lognormal(13, 0.3, size=(n_days, len(syms))), index=idx, columns=syms
    )
    return close, volume


def test_attribution_sums_to_raw_and_score():
    close, volume = _panel()
    asof = close.index[-1]
    out = S.compute_all(close, volume, asof)

    for name in ("capital_rotation", "risk_on_off", "growth", "inflation",
                 "recession", "liquidity"):
        sig = out[name]
        assert sig["score"] is not None, f"{name} produced no score on the test panel"
        attr = sig.get("attribution")
        assert attr, f"{name} missing attribution"
        drivers = attr["drivers"]
        assert drivers, f"{name} attribution has no drivers"
        # Contributions must sum to the pre-mapping aggregate…
        assert sum(d["contrib"] for d in drivers) == pytest.approx(attr["raw"], abs=1e-9)
        # …and the report's Pts rescale must sum to score − baseline.
        span = sig["score"] - attr["baseline"]
        if attr["raw"]:
            pts = [d["contrib"] / attr["raw"] * span for d in drivers]
            assert sum(pts) == pytest.approx(span, abs=1e-9)
        # Sorted by |contrib| descending.
        mags = [abs(d["contrib"]) for d in drivers]
        assert mags == sorted(mags, reverse=True)


def test_attribution_baselines():
    close, volume = _panel(seed=11)
    asof = close.index[-1]
    out = S.compute_all(close, volume, asof)
    assert out["liquidity"]["attribution"]["baseline"] == 50.0
    assert out["growth"]["attribution"]["baseline"] == 0.0


def test_section_signal_attribution_renders_and_falls_back():
    signals = {
        "growth": {
            "score": -45.8,
            "attribution": {
                "drivers": [
                    {"driver": "SMH", "value": -1.2, "weight": 0.2, "contrib": -0.24},
                    {"driver": "XLU", "value": 0.4, "weight": -0.33, "contrib": -0.13},
                ],
                "raw": -0.37,
                "baseline": 0.0,
            },
        },
        # Legacy row computed before attribution shipped.
        "recession": {"score": 12.0},
        # No score at all -> skipped entirely.
        "liquidity": {"score": None},
    }
    md = section_signal_attribution(signals, n=21)
    assert "## Section 21 — Signal Attribution" in md
    assert "### Growth: -45.8" in md
    assert "| SMH |" in md and "| XLU |" in md
    # Pts column sums to the headline: total row shows the score.
    assert "**-45.8**" in md
    assert "### Recession: +12.0" in md
    assert "Re-run `rotate signals`" in md
    assert "Liquidity" not in md


# ============================================================
# W1 — forecast scorecard
# ============================================================

def _seed_spy_bars(cfg: Config, start: date, n_days: int, drift: float) -> list[date]:
    """Business-day SPY bars with constant log drift. Returns the bar dates."""
    rows, dates = [], []
    d, i = start, 0
    px = 500.0
    while i < n_days:
        if d.weekday() < 5:
            px *= float(np.exp(drift))
            rows.append({
                "symbol": "SPY", "asset_class": "equity_index", "ts": d,
                "open": px, "high": px * 1.01, "low": px * 0.99,
                "close": px, "adj_close": px, "volume": 1e6,
                "source": "test", "ingested_at": datetime(2026, 1, 1),
                "stale": False,
            })
            dates.append(d)
            i += 1
        d += timedelta(days=1)
    with connect(cfg.storage.duckdb_path) as con:
        upsert_bars(con, rows)
    return dates


def _insert_forecast(cfg: Config, ts: date, ftype: str, horizon: int, **kw):
    defaults = dict(target="SPY", direction="bullish", bullish_pct=0.7,
                    neutral_pct=0.2, bearish_pct=0.1, median_fwd=0.004,
                    cutoff=0.005, confidence=0.6, n_analogues=30, details=None)
    defaults.update(kw)
    with connect(cfg.storage.duckdb_path) as con:
        con.execute(
            "INSERT INTO forecasts (ts, forecast_type, horizon_days, target, "
            "direction, bullish_pct, neutral_pct, bearish_pct, median_fwd, cutoff, "
            "confidence, n_analogues, details, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [ts, ftype, horizon, defaults["target"], defaults["direction"],
             defaults["bullish_pct"], defaults["neutral_pct"], defaults["bearish_pct"],
             defaults["median_fwd"], defaults["cutoff"], defaults["confidence"],
             defaults["n_analogues"], defaults["details"], datetime(2026, 1, 1)],
        )


def test_resolve_spy_forecast_hit_and_miss(tmp_path):
    cfg = _cfg(tmp_path)
    # Strong positive drift: 5d forward log return = 5 * 0.004 = 2% >> cutoff.
    dates = _seed_spy_bars(cfg, date(2026, 1, 5), 30, drift=0.004)
    f_ts = dates[0]
    _insert_forecast(cfg, f_ts, "spy_5d", 5, direction="bullish")
    _insert_forecast(cfg, f_ts, "spy_21d", 21, direction="bearish")

    out = SC.resolve_forecasts(cfg, dates[-1])
    assert out["resolved"] == 2

    with connect(cfg.storage.duckdb_path) as con:
        df = con.execute(
            "SELECT forecast_type, actual_value, actual_direction, hit "
            "FROM forecasts ORDER BY forecast_type"
        ).df().set_index("forecast_type")
    assert df.loc["spy_5d", "actual_direction"] == "bullish"
    assert bool(df.loc["spy_5d", "hit"]) is True
    assert df.loc["spy_5d", "actual_value"] == pytest.approx(5 * 0.004, rel=1e-6)
    # 21d realized bullish, forecast said bearish -> miss.
    assert df.loc["spy_21d", "actual_direction"] == "bullish"
    assert bool(df.loc["spy_21d", "hit"]) is False


def test_resolve_waits_for_horizon(tmp_path):
    cfg = _cfg(tmp_path)
    dates = _seed_spy_bars(cfg, date(2026, 1, 5), 10, drift=0.004)
    _insert_forecast(cfg, dates[-3], "spy_5d", 5)  # only 2 bars after publish
    out = SC.resolve_forecasts(cfg, dates[-1])
    assert out["resolved"] == 0
    with connect(cfg.storage.duckdb_path) as con:
        pending = con.execute(
            "SELECT COUNT(*) FROM forecasts WHERE resolved_at IS NULL"
        ).fetchone()[0]
    assert pending == 1


def test_resolve_sector_forecast(tmp_path):
    cfg = _cfg(tmp_path)
    f_ts = date(2026, 1, 5)
    # 22 business days of metrics after f_ts: SMH strongly up (Technology),
    # TLT strongly down (Long Bonds), XLV mild (Healthcare).
    rows = []
    d, i = f_ts + timedelta(days=1), 0
    while i < 22:
        if d.weekday() < 5:
            for sym, rd in (("SMH", 0.01), ("TLT", -0.01), ("XLV", 0.001)):
                rows.append((d, sym, rd))
            i += 1
        d += timedelta(days=1)
    with connect(cfg.storage.duckdb_path) as con:
        for ts, sym, rd in rows:
            con.execute(
                "INSERT INTO metrics_daily (ts, symbol, r_d, computed_at) VALUES (?,?,?,?)",
                [ts, sym, rd, datetime(2026, 1, 1)],
            )
    _insert_forecast(
        cfg, f_ts, "sector_21d", 21, target=None, direction=None,
        bullish_pct=None, neutral_pct=None, bearish_pct=None, median_fwd=None,
        cutoff=None, confidence=None, n_analogues=None,
        details=json.dumps({"predicted_top3": ["Technology", "Energy", "Crypto"]}),
    )
    out = SC.resolve_forecasts(cfg, date(2026, 3, 1))
    assert out["resolved"] == 1
    with connect(cfg.storage.duckdb_path) as con:
        row = con.execute("SELECT hit, details FROM forecasts").fetchone()
    assert bool(row[0]) is True  # predicted #1 (Technology) in realized top-3
    det = json.loads(row[1])
    assert det["actual_top3"][0] == "Technology"


def test_record_forecasts_skips_resolved_rows(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    f_ts = date(2026, 1, 5)
    _insert_forecast(cfg, f_ts, "spy_5d", 5, direction="bearish")
    with connect(cfg.storage.duckdb_path) as con:
        con.execute(
            "UPDATE forecasts SET resolved_at = ?, hit = TRUE WHERE ts = ?",
            [datetime(2026, 2, 1), f_ts],
        )

    # Canned analogue outputs so record_forecasts produces candidate rows.
    monkeypatch.setattr(SC, "forecast_distribution", lambda *a, **k: {
        "target": "SPY", "horizon": k.get("horizon", 5), "cutoff": 0.005,
        "bullish_pct": 0.7, "neutral_pct": 0.2, "bearish_pct": 0.1,
        "mean_fwd": 0.004, "median_fwd": 0.004, "p10_fwd": -0.01, "p90_fwd": 0.02,
        "n_analogues": 30, "top_similarity": 0.9, "agreement": 0.7,
    })
    monkeypatch.setattr(SC, "sector_leader_forecast",
                        lambda *a, **k: pd.DataFrame())
    monkeypatch.setattr(SC, "latest_verdicts", lambda cfg: {})

    out = SC.record_forecasts(cfg, f_ts)
    # spy_5d already resolved -> immutable; spy_21d is new.
    assert out["recorded"] == 1
    with connect(cfg.storage.duckdb_path) as con:
        row = con.execute(
            "SELECT direction, hit FROM forecasts WHERE forecast_type = 'spy_5d'"
        ).fetchone()
    assert row[0] == "bearish" and bool(row[1]) is True  # untouched


def test_scorecard_section_renders(tmp_path):
    cfg = _cfg(tmp_path)
    dates = _seed_spy_bars(cfg, date(2026, 1, 5), 30, drift=0.004)
    _insert_forecast(cfg, dates[0], "spy_5d", 5, direction="bullish")
    SC.resolve_forecasts(cfg, dates[-1])

    with connect(cfg.storage.duckdb_path) as con:
        md = section_forecast_scorecard(con, dates[-1], n=14)
    assert "## Section 14 — Forecast Scorecard — Actual vs Forecast" in md
    assert "**100%**" in md          # 1/1 hit
    assert "Bullish (70%)" in md     # recent-history row
    assert "✓" in md


def test_scorecard_section_empty_store(tmp_path):
    cfg = _cfg(tmp_path)
    with connect(cfg.storage.duckdb_path) as con:
        md = section_forecast_scorecard(con, date(2026, 6, 9), n=14)
    assert "No forecasts recorded yet" in md


# ============================================================
# W2 — Investment Committee View (US)
# ============================================================

def test_committee_view_bullets_and_lean(tmp_path):
    cfg = _cfg(tmp_path)
    with connect(cfg.storage.duckdb_path):
        pass  # schema only — no analogue history, so qualitative-lean fallback

    signals = {
        "capital_rotation": {"score": 32.0},
        "risk_on_off": {"score": -40.0},
        "growth": {"score": 20.0},
        "inflation": {"score": 30.0},
        "recession": {"score": 25.0},
        "liquidity": {"score": 35.0},
    }
    metrics = pd.DataFrame({
        "symbol": ["XLF", "XLV", "SMH", "XLK"],
        "r_d": [0.001, 0.002, -0.004, -0.003],
        "r_w": [0.012, 0.008, -0.015, -0.011],
        "r_m": [0.03, 0.02, -0.04, -0.03],
    })
    regime = {"regime": "Risk-Off Contraction", "days_in_regime": 4}

    md = section_committee_view(cfg, date(2026, 6, 9), signals, regime, None, metrics, n=2)
    assert "## Section 2 — Investment Committee View" in md
    assert "### Bull Case" in md and "### Bear Case" in md
    assert "Capital rotating toward risk assets (rotation +32)" in md
    assert "Risk-off tone across baskets (risk-on/off -40)" in md
    assert "Inflation pressure building" in md
    assert "Recession concern elevated (+25)" in md
    assert "Liquidity tightening (35/100)" in md
    assert "Financials attracting capital" in md
    assert "Money leaving Technology" in md
    # No analogue history -> qualitative lean, never invented probabilities.
    assert "qualitative lean" in md.lower()
    assert "### Suggested Positioning" in md
    assert "Overweight Financials" in md
    assert "Underweight Technology" in md
    assert "elevated cash buffer" in md
    assert "not investment advice" in md


def test_committee_view_flat_market(tmp_path):
    cfg = _cfg(tmp_path)
    with connect(cfg.storage.duckdb_path):
        pass
    signals = {"capital_rotation": {"score": 2.0}, "liquidity": {"score": 50.0}}
    md = section_committee_view(cfg, date(2026, 6, 9), signals, None, None,
                                pd.DataFrame(), n=2)
    assert "no signal crosses its bull threshold" in md
    assert "no signal crosses its bear threshold" in md
    assert "Normal cash allocation" in md
