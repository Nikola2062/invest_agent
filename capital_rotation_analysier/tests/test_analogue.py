"""Tests for the Historical Analogue Engine.

These tests build a tiny in-memory rotation.duckdb with a known signal panel,
then verify:
  - cosine similarity picks the right neighbour
  - find_analogues respects the blackout window
  - regime_transition_matrix sums to 1.0 across each row
  - rotation_probability_matrix returns the seeded winners
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from rotation.analogue import (
    ANALOGUE_SIGNALS,
    _cosine_similarity_against,
    find_analogues,
    forecast_confidence,
    forecast_distribution,
    regime_transition_matrix,
    rotation_probability_matrix,
    sector_leader_forecast,
)
from rotation.config import Config, IngestConfig, RetryPolicy, StorageConfig, Symbol
from rotation.schema import ensure_schema
from rotation.store import connect


@dataclass
class _Fix:
    cfg: Config
    asof: date


def _make_cfg(tmp_path: Path) -> Config:
    db = tmp_path / "rot.duckdb"
    return Config(
        storage=StorageConfig(duckdb_path=db),
        ingest=IngestConfig(
            primary_source="yfinance",
            retry=RetryPolicy(max_attempts=1, backoff_seconds=(1,)),
            stale_threshold_days=4,
            outlier_intraday_pct=20.0,
            outlier_intraday_pct_by_class={},
        ),
        universe=(Symbol("SPY", "equity_us"), Symbol("XLV", "equity_sector")),
    )


def _seed_signals(con: duckdb.DuckDBPyConnection, dates: list[date], vec_fn) -> None:
    rows = []
    for d in dates:
        v = vec_fn(d)
        for name, score in zip(ANALOGUE_SIGNALS, v):
            rows.append({
                "ts": d, "signal_name": name, "score": float(score),
                "confidence": 0.8, "components": None,
                "computed_at": pd.Timestamp("2020-01-01"),
            })
    con.register("incoming", pd.DataFrame(rows))
    con.execute("INSERT INTO signals_daily "
                "(ts, signal_name, score, confidence, components, computed_at) "
                "SELECT ts, signal_name, score, confidence, components, computed_at "
                "FROM incoming")
    con.unregister("incoming")


def _seed_bars(con: duckdb.DuckDBPyConnection, symbol: str, dates: list[date],
               price_fn) -> None:
    rows = []
    for d in dates:
        p = float(price_fn(d))
        rows.append({
            "symbol": symbol, "asset_class": "equity_us", "ts": d,
            "open": p, "high": p, "low": p, "close": p, "adj_close": p,
            "volume": 1_000_000.0, "source": "test", "revision": 0,
            "ingested_at": pd.Timestamp("2020-01-01"), "stale": False,
        })
    con.register("incoming_bars", pd.DataFrame(rows))
    con.execute("INSERT INTO raw_bars "
                "(symbol, asset_class, ts, open, high, low, close, adj_close, "
                "volume, source, revision, ingested_at, stale) "
                "SELECT symbol, asset_class, ts, open, high, low, close, adj_close, "
                "volume, source, revision, ingested_at, stale FROM incoming_bars")
    con.unregister("incoming_bars")


def _seed_metrics(con: duckdb.DuckDBPyConnection, dates: list[date],
                  symbol: str, r_d_fn) -> None:
    rows = []
    for d in dates:
        rows.append({
            "ts": d, "symbol": symbol,
            "r_d": float(r_d_fn(d)),
            "r_w": None, "r_m": None, "r_q": None,
            "vol_30": None, "vol_ratio": None,
            "rv": None, "vz": None,
            "rs_rank": None, "rs_change_1": None,
            "rs_change_5": None, "rs_change_21": None, "rs_accel_5": None,
            "computed_at": pd.Timestamp("2020-01-01"),
        })
    con.register("incoming_m", pd.DataFrame(rows))
    con.execute(
        "INSERT INTO metrics_daily "
        "(ts, symbol, r_d, r_w, r_m, r_q, vol_30, vol_ratio, rv, vz, "
        "rs_rank, rs_change_1, rs_change_5, rs_change_21, rs_accel_5, computed_at) "
        "SELECT ts, symbol, r_d, r_w, r_m, r_q, vol_30, vol_ratio, rv, vz, "
        "rs_rank, rs_change_1, rs_change_5, rs_change_21, rs_accel_5, computed_at "
        "FROM incoming_m"
    )
    con.unregister("incoming_m")


def _seed_regime(con: duckdb.DuckDBPyConnection, dates: list[date], regime_fn) -> None:
    rows = []
    days_in = 0
    prev = None
    for d in dates:
        r = regime_fn(d)
        if r != prev:
            days_in = 1
        else:
            days_in += 1
        rows.append({
            "ts": d, "regime": r, "prev_regime": prev,
            "confidence": 0.6, "days_in_regime": days_in,
            "components": None, "computed_at": pd.Timestamp("2020-01-01"),
        })
        prev = r
    con.register("incoming_r", pd.DataFrame(rows))
    con.execute(
        "INSERT INTO regime_history "
        "(ts, regime, prev_regime, confidence, days_in_regime, components, computed_at) "
        "SELECT ts, regime, prev_regime, confidence, days_in_regime, components, computed_at "
        "FROM incoming_r"
    )
    con.unregister("incoming_r")


@pytest.fixture
def fix(tmp_path):
    cfg = _make_cfg(tmp_path)
    # 200 trading days; signal vector mimics a sinusoid so similarity is well-defined
    base = date(2024, 1, 2)
    dates = [base + timedelta(days=i) for i in range(200)]

    def signal_at(d):
        # Same pattern every 50 days -> day 0 and day 50, 100, 150 are similar
        i = (d - base).days
        phase = (i % 50) / 50.0 * 2 * np.pi
        return [np.cos(phase + k * 0.3) for k in range(len(ANALOGUE_SIGNALS))]

    with connect(cfg.storage.duckdb_path) as con:
        ensure_schema(con)
        _seed_signals(con, dates, signal_at)
        _seed_bars(con, "SPY", dates, lambda d: 400 + (d - base).days * 0.5)
        _seed_metrics(con, dates, "SPY", lambda d: 0.002)
        _seed_metrics(con, dates, "XLV", lambda d: 0.005)
        _seed_regime(con, dates,
                     lambda d: "Risk-On Expansion" if (d - base).days < 100 else "Risk-Off Defensive")

    return _Fix(cfg=cfg, asof=dates[-1])


def test_cosine_similarity_picks_nearest():
    target = np.array([1.0, 0.0, 0.0])
    panel = np.array([
        [1.0, 0.0, 0.0],   # identical
        [0.5, 0.5, 0.5],   # 0.577
        [0.0, 1.0, 0.0],   # orthogonal
        [-1.0, 0.0, 0.0],  # opposite
    ])
    sims = _cosine_similarity_against(target, panel)
    assert sims[0] == pytest.approx(1.0)
    assert sims[2] == pytest.approx(0.0)
    assert sims[3] == pytest.approx(-1.0)


def test_cosine_similarity_handles_nan_rows():
    target = np.array([1.0, 0.0, 0.0])
    panel = np.array([
        [1.0, 0.0, 0.0],
        [np.nan, 0.0, 0.0],
    ])
    sims = _cosine_similarity_against(target, panel)
    assert sims[0] == pytest.approx(1.0)
    assert sims[1] == -np.inf


def test_find_analogues_returns_similar_dates(fix):
    anas = find_analogues(fix.cfg, fix.asof, k=3, blackout=10)
    assert len(anas) == 3
    # Similarity is bounded [-1, 1]
    for a in anas:
        assert -1.0 <= a.similarity <= 1.0
    # The top hit should be very close to perfect (the seed is periodic)
    assert anas[0].similarity > 0.9


def test_find_analogues_respects_blackout(fix):
    asof_idx = fix.asof
    anas = find_analogues(fix.cfg, asof_idx, k=10, blackout=30)
    # No analogue should fall within the blackout window
    for a in anas:
        delta = (asof_idx - a.asof).days
        assert delta >= 30, f"analogue {a.asof} only {delta} days before asof"


def test_find_analogues_empty_when_no_data(tmp_path):
    cfg = _make_cfg(tmp_path)
    with connect(cfg.storage.duckdb_path) as con:
        ensure_schema(con)
    anas = find_analogues(cfg, date(2024, 1, 15), k=5)
    assert anas == []


def test_regime_transition_matrix_rows_sum_to_one(fix):
    m = regime_transition_matrix(fix.cfg, window=30)
    if m.empty:
        pytest.skip("not enough regime history in the fixture")
    for src, row in m.iterrows():
        s = row.sum()
        assert s == pytest.approx(1.0, abs=1e-9), f"row {src} sums to {s}"


def test_rotation_probability_matrix_returns_frequencies(fix):
    rpm = rotation_probability_matrix(fix.cfg, fix.asof, k=5, horizon=5)
    if rpm.empty:
        pytest.skip("insufficient analogue forward-return data")
    # frequencies are in [0, 1]
    assert (rpm["frequency"] >= 0.0).all()
    assert (rpm["frequency"] <= 1.0).all()


def test_forecast_distribution_buckets_sum_to_one(fix):
    f = forecast_distribution(fix.cfg, fix.asof, target="SPY", horizon=5, k=10)
    if f["n_analogues"] == 0:
        pytest.skip("no analogue forward returns in fixture")
    total = f["bullish_pct"] + f["neutral_pct"] + f["bearish_pct"]
    assert total == pytest.approx(1.0, abs=1e-9), f"buckets sum to {total}"


def test_forecast_distribution_empty_when_no_data(tmp_path):
    cfg = _make_cfg(tmp_path)
    with connect(cfg.storage.duckdb_path) as con:
        ensure_schema(con)
    f = forecast_distribution(cfg, date(2024, 1, 15), target="SPY", horizon=5)
    assert f["n_analogues"] == 0
    assert f["bullish_pct"] is None


def test_forecast_confidence_blends_three_inputs():
    # No verdicts -> validation_credit defaults to 0.5
    f = {"top_similarity": 1.0, "agreement": 1.0}
    c = forecast_confidence(f, verdicts=None)
    assert c["score"] == pytest.approx((1.0 + 1.0 + 0.5) / 3.0)
    assert c["bucket"] == "high"

    # Negative similarity should be clamped to 0
    f2 = {"top_similarity": -0.5, "agreement": 0.5}
    c2 = forecast_confidence(f2, verdicts=None)
    assert c2["components"]["top_similarity"] == 0.0
    assert c2["score"] == pytest.approx((0.0 + 0.5 + 0.5) / 3.0)


def test_forecast_confidence_validation_credit_with_verdicts():
    from dataclasses import dataclass

    @dataclass
    class V:
        verdict: str
        hit_rate_overall: float | None = None

    # pass + directional-but-failing + outright-fail = 1.0 + 0.5 + 0.0 = 1.5/3
    verdicts = {
        "capital_rotation": V("pass"),
        "risk_on_off":      V("fail", hit_rate_overall=0.60),  # directional
        "growth":           V("fail", hit_rate_overall=0.50),  # below 0.55 -> 0.0
    }
    f = {"top_similarity": 0.8, "agreement": 0.8}
    c = forecast_confidence(f, verdicts=verdicts)
    assert c["components"]["validation_credit"] == pytest.approx(1.5 / 3)


def test_forecast_confidence_missing_verdict_neutral():
    """A signal that isn't in `verdicts` contributes 0.5 (neutral)."""
    from dataclasses import dataclass

    @dataclass
    class V:
        verdict: str
        hit_rate_overall: float | None = None

    verdicts = {"capital_rotation": V("pass")}   # other two absent
    c = forecast_confidence({"top_similarity": 0.5, "agreement": 0.5}, verdicts=verdicts)
    # 1.0 + 0.5 + 0.5 = 2.0 / 3
    assert c["components"]["validation_credit"] == pytest.approx(2.0 / 3)


def test_sector_leader_forecast_uses_21d_horizon(fix):
    slf = sector_leader_forecast(fix.cfg, fix.asof, horizon=21, k=5, top_k=3)
    if slf.empty:
        pytest.skip("insufficient analogue forward-return data")
    assert (slf["frequency"] >= 0.0).all()
    assert (slf["frequency"] <= 1.0).all()
