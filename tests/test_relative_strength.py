"""Unit tests for the cross-sectional relative-strength core (Phase 3).

Pure functions only — every price panel is synthetic, no network. Mirrors the
style of tests/test_sizing.py and tests/test_position_overlay.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.portfolio import (
    coarse_regime,
    rank,
    relative_strength_score,
)
from tradingagents.portfolio.relative_strength import (
    REGIME_NEUTRAL,
    REGIME_RISK_OFF,
    REGIME_RISK_ON,
)

pytestmark = pytest.mark.unit


# --- helpers --------------------------------------------------------------


def _trend_panel(daily_drifts, n=200, start="2023-01-01"):
    """Deterministic price panel; each column compounds at a fixed daily drift.

    ``daily_drifts`` maps ticker -> per-bar return (e.g. 0.001 = +0.1%/day).
    No noise, so relative strength is fully determined by the drift.
    """
    idx = pd.bdate_range(start, periods=n)
    cols = {
        tkr: 100.0 * (1.0 + d) ** np.arange(n)
        for tkr, d in daily_drifts.items()
    }
    return pd.DataFrame(cols, index=idx)


# --- relative_strength_score ----------------------------------------------


def test_outperformer_scores_positive_laggard_negative():
    prices = _trend_panel({"LEAD": 0.002, "LAG": -0.001, "BENCH": 0.0005})
    lead = relative_strength_score(prices["LEAD"], prices["BENCH"])
    lag = relative_strength_score(prices["LAG"], prices["BENCH"])
    assert lead > 0
    assert lag < 0
    assert lead > lag


def test_score_zero_when_matching_benchmark():
    prices = _trend_panel({"A": 0.001, "BENCH": 0.001})
    assert relative_strength_score(prices["A"], prices["BENCH"]) == pytest.approx(0.0, abs=1e-9)


def test_short_series_returns_zero_without_raising():
    # 10 bars: shorter than every default lookback (21/63/126) -> neutral 0.0.
    prices = _trend_panel({"A": 0.01, "BENCH": 0.0}, n=10)
    score = relative_strength_score(prices["A"], prices["BENCH"])
    assert score == 0.0


def test_uses_only_usable_lookbacks():
    # 30 bars: only the 21-day window is usable; longer ones are skipped, no raise.
    prices = _trend_panel({"A": 0.003, "BENCH": 0.0}, n=30)
    score = relative_strength_score(prices["A"], prices["BENCH"])
    assert score > 0  # outperformed on the one usable window


# --- rank -----------------------------------------------------------------


def test_rank_orders_best_first_and_excludes_benchmark():
    prices = _trend_panel({
        "BEST": 0.003,
        "MID": 0.001,
        "WORST": -0.002,
        "BENCH": 0.0005,
    })
    ranked = rank(prices, benchmark_col="BENCH")
    symbols = [r["symbol"] for r in ranked]
    assert "BENCH" not in symbols          # benchmark excluded
    assert symbols == ["BEST", "MID", "WORST"]
    assert [r["rank"] for r in ranked] == [1, 2, 3]
    assert ranked[0]["rs_score"] > ranked[-1]["rs_score"]


def test_rank_trailing_returns_populated():
    prices = _trend_panel({"A": 0.002, "BENCH": 0.0005})
    ranked = rank(prices, benchmark_col="BENCH")
    tr = ranked[0]["trailing_returns"]
    assert set(tr) == {21, 63, 126}        # all default lookbacks usable on 200 bars
    assert all(v > 0 for v in tr.values())  # positive-drift name has positive returns


def test_rank_respects_explicit_ticker_subset():
    prices = _trend_panel({"A": 0.002, "B": 0.001, "C": -0.001, "BENCH": 0.0})
    ranked = rank(prices, benchmark_col="BENCH", tickers=["A", "C"])
    assert {r["symbol"] for r in ranked} == {"A", "C"}


def test_rank_missing_benchmark_raises():
    prices = _trend_panel({"A": 0.001, "BENCH": 0.0})
    with pytest.raises(KeyError):
        rank(prices, benchmark_col="NOPE")


# --- coarse_regime --------------------------------------------------------


def test_regime_risk_on_when_breadth_high_and_bench_rising():
    # 4 names all beating a modestly-rising benchmark -> 100% breadth, bench up.
    prices = _trend_panel({
        "A": 0.003, "B": 0.0025, "C": 0.002, "D": 0.0018,
        "BENCH": 0.0005,
    })
    assert coarse_regime(prices, benchmark_col="BENCH") == REGIME_RISK_ON


def test_regime_risk_off_when_breadth_low_and_bench_falling():
    # Falling benchmark, all names lagging it -> Risk-Off Defensive (exact string).
    prices = _trend_panel({
        "A": -0.004, "B": -0.0035, "C": -0.003, "D": -0.0025,
        "BENCH": -0.001,
    })
    regime = coarse_regime(prices, benchmark_col="BENCH")
    assert regime == REGIME_RISK_OFF
    assert regime == "Risk-Off Defensive"  # must match OverlayConfig.risk_off_regimes


def test_regime_neutral_when_mixed():
    # Half the names beat a flat-ish benchmark, benchmark essentially flat.
    prices = _trend_panel({
        "A": 0.0008, "B": 0.0007, "C": -0.0003, "D": -0.0004,
        "BENCH": 0.0002,
    })
    assert coarse_regime(prices, benchmark_col="BENCH") == REGIME_NEUTRAL


def test_regime_neutral_on_short_panel():
    # Too short to compute a 63-day trailing return -> safe Neutral default.
    prices = _trend_panel({"A": 0.01, "B": -0.01, "BENCH": 0.0}, n=20)
    assert coarse_regime(prices, benchmark_col="BENCH") == REGIME_NEUTRAL
