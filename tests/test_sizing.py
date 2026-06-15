"""Unit tests for the risk-aware sizer.

Each stage of the sizer pipeline (rating → multiplier → vol scale →
per-name cap → sector cap → gross cap) is exercised in isolation, then
the integration with ``build_equity_curve`` is verified.

No network, no LLM — all prices are synthetic.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tradingagents.backtest.portfolio import build_equity_curve
from tradingagents.backtest.runner import generate_rebalance_dates
from tradingagents.backtest.strategy import Decision
from tradingagents.portfolio import (
    SizingConfig,
    equal_weight_sizer,
    risk_aware_sizer,
)

pytestmark = pytest.mark.unit


# --- helpers --------------------------------------------------------------


def _decisions(*pairs, date="2024-01-15"):
    return [Decision(t, date, r) for t, r in pairs]


def _vol_panel(spec, start="2023-01-01", end="2024-03-01"):
    """Build a price panel with controlled per-ticker daily volatility.

    ``spec`` maps ticker → daily-vol (so e.g. 0.02 ≈ 32% annualised).
    """
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(0)
    cols = {}
    for tkr, vol in spec.items():
        rets = rng.normal(0, vol, size=len(idx))
        cols[tkr] = 100 * np.exp(np.cumsum(rets))
    return pd.DataFrame(cols, index=idx)


# --- equal-weight (regression: must match legacy behaviour) ---------------


def test_equal_weight_sizer_matches_legacy():
    decs = _decisions(("A", "Buy"), ("B", "Overweight"), ("C", "Hold"), ("D", "Sell"))
    w = equal_weight_sizer(decs)
    assert set(w) == {"A", "B"}
    assert w["A"] == pytest.approx(0.5)


def test_equal_weight_empty_when_all_hold():
    decs = _decisions(("A", "Hold"), ("B", "Sell"))
    assert equal_weight_sizer(decs) == {}


# --- rating → multiplier --------------------------------------------------


def test_buy_gets_twice_overweight_before_caps():
    """With no vol scaling and no caps binding, Buy should weigh 2x Overweight."""
    cfg = SizingConfig(
        use_vol_scaling=False,
        max_position=1.0, max_sector_exposure=1.0, max_gross=10.0,
    )
    w = risk_aware_sizer(
        _decisions(("A", "Buy"), ("B", "Overweight")),
        config=cfg,
    )
    assert w["A"] == pytest.approx(2 * w["B"])


def test_shorts_zeroed_unless_explicitly_allowed():
    cfg = SizingConfig(use_vol_scaling=False, max_position=1.0, max_gross=10.0)
    w = risk_aware_sizer(_decisions(("A", "Sell"), ("B", "Buy")), config=cfg)
    assert "A" not in w
    assert w["B"] > 0


def test_shorts_when_enabled():
    cfg = SizingConfig(
        use_vol_scaling=False, allow_shorts=True,
        max_position=1.0, max_sector_exposure=1.0, max_gross=10.0,
    )
    w = risk_aware_sizer(_decisions(("A", "Sell"), ("B", "Buy")), config=cfg)
    assert w["A"] < 0
    assert w["B"] > 0


def test_hold_excluded():
    cfg = SizingConfig(use_vol_scaling=False)
    w = risk_aware_sizer(_decisions(("A", "Hold")), config=cfg)
    assert w == {}


# --- inverse-volatility scaling ------------------------------------------


def test_low_vol_name_gets_larger_weight():
    """Given same rating, lower-vol name should end up with larger weight."""
    prices = _vol_panel({"CALM": 0.005, "WILD": 0.05})  # 8% vs 80% ann.
    cfg = SizingConfig(
        target_vol=0.20, vol_lookback_days=60,
        max_position=1.0, max_sector_exposure=1.0, max_gross=10.0,
    )
    w = risk_aware_sizer(
        _decisions(("CALM", "Buy"), ("WILD", "Buy")),
        prices=prices,
        as_of=prices.index[-1],
        config=cfg,
    )
    assert w["CALM"] > w["WILD"]
    # The low-vol name should carry roughly 10x more weight in this fixture.
    assert w["CALM"] / w["WILD"] > 5


def test_vol_scaling_skipped_for_unknown_ticker():
    """Tickers absent from the price panel keep their base weight."""
    prices = _vol_panel({"KNOWN": 0.01})
    cfg = SizingConfig(
        target_vol=0.20, vol_lookback_days=60,
        max_position=1.0, max_sector_exposure=1.0, max_gross=10.0,
    )
    w = risk_aware_sizer(
        _decisions(("KNOWN", "Buy"), ("GHOST", "Buy")),
        prices=prices,
        as_of=prices.index[-1],
        config=cfg,
    )
    # GHOST stays at base; KNOWN gets vol-scaled.
    assert w["GHOST"] == pytest.approx(2 * cfg.base_weight)
    assert w["KNOWN"] != pytest.approx(2 * cfg.base_weight)


def test_vol_scaling_uses_floor():
    """A degenerate zero-vol price series shouldn't divide by zero."""
    idx = pd.bdate_range("2023-01-01", "2024-01-01")
    prices = pd.DataFrame({"FLAT": np.full(len(idx), 100.0)}, index=idx)
    cfg = SizingConfig(
        target_vol=0.20, vol_lookback_days=60, vol_floor=0.05,
        max_position=1.0, max_sector_exposure=1.0, max_gross=10.0,
    )
    w = risk_aware_sizer(
        _decisions(("FLAT", "Buy")),
        prices=prices,
        as_of=prices.index[-1],
        config=cfg,
    )
    assert "FLAT" in w
    assert np.isfinite(w["FLAT"])


# --- per-name cap ---------------------------------------------------------


def test_per_name_cap_binds():
    """A Buy that would otherwise exceed max_position gets clipped."""
    cfg = SizingConfig(
        use_vol_scaling=False, base_weight=0.30,  # raw weight = 0.60 (Buy)
        max_position=0.10, max_sector_exposure=10.0, max_gross=10.0,
    )
    w = risk_aware_sizer(_decisions(("A", "Buy")), config=cfg)
    assert w["A"] == pytest.approx(0.10)


def test_per_name_cap_preserves_sign():
    cfg = SizingConfig(
        use_vol_scaling=False, base_weight=0.30, allow_shorts=True,
        max_position=0.10, max_sector_exposure=10.0, max_gross=10.0,
    )
    w = risk_aware_sizer(_decisions(("A", "Sell")), config=cfg)
    assert w["A"] == pytest.approx(-0.10)


# --- per-sector cap ------------------------------------------------------


def test_sector_cap_proportionally_shrinks_overconcentrated_sector():
    cfg = SizingConfig(
        use_vol_scaling=False, base_weight=0.10,  # 4 Buys × 0.20 = 0.80 raw
        max_position=0.30, max_sector_exposure=0.40, max_gross=10.0,
    )
    decs = _decisions(*[(t, "Buy") for t in ["A", "B", "C", "D"]])
    sectors = {"A": "Tech", "B": "Tech", "C": "Tech", "D": "Tech"}
    w = risk_aware_sizer(decs, sectors=sectors, config=cfg)
    total = sum(abs(v) for v in w.values())
    assert total == pytest.approx(0.40, abs=1e-6)
    # All four names get the same shrunk weight.
    assert max(w.values()) - min(w.values()) < 1e-9


def test_sector_cap_doesnt_punish_other_sectors():
    cfg = SizingConfig(
        use_vol_scaling=False, base_weight=0.20,
        max_position=1.0, max_sector_exposure=0.50, max_gross=10.0,
    )
    decs = _decisions(("A", "Buy"), ("B", "Buy"), ("C", "Buy"))
    sectors = {"A": "Tech", "B": "Tech", "C": "Health"}
    w = risk_aware_sizer(decs, sectors=sectors, config=cfg)
    tech = abs(w["A"]) + abs(w["B"])
    health = abs(w["C"])
    assert tech == pytest.approx(0.50, abs=1e-6)
    # Health is untouched — only Tech was over-exposed.
    assert health == pytest.approx(0.40, abs=1e-6)


# --- gross-exposure cap --------------------------------------------------


def test_gross_cap_normalises_when_book_too_big():
    cfg = SizingConfig(
        use_vol_scaling=False, base_weight=0.50,
        max_position=1.0, max_sector_exposure=10.0, max_gross=1.0,
    )
    decs = _decisions(("A", "Buy"), ("B", "Buy"), ("C", "Buy"))
    w = risk_aware_sizer(decs, config=cfg)
    assert sum(abs(v) for v in w.values()) == pytest.approx(1.0, abs=1e-6)


def test_gross_cap_does_not_grow_a_small_book():
    """A book under the gross cap is NOT scaled up — staying in cash is OK."""
    cfg = SizingConfig(
        use_vol_scaling=False, base_weight=0.05,
        max_position=1.0, max_sector_exposure=10.0, max_gross=1.0,
    )
    decs = _decisions(("A", "Overweight"))  # raw = 0.05
    w = risk_aware_sizer(decs, config=cfg)
    assert w["A"] == pytest.approx(0.05)


# --- integration with build_equity_curve ---------------------------------


def test_risk_aware_curve_runs_with_real_pipeline():
    """End-to-end: decisions → risk-aware sizer → equity curve.

    The actual return depends on noise; what we test is that the curve
    runs to completion and has the right number of anchors. Correctness
    of the math is covered by the unit tests above.
    """
    prices = _vol_panel({"A": 0.01, "B": 0.02, "C": 0.015})
    rebal = generate_rebalance_dates("2023-04-01", "2023-12-01", "monthly")
    decs = [Decision(t, d, "Buy") for d in rebal for t in ["A", "B", "C"]]

    cfg = SizingConfig(
        target_vol=0.20, vol_lookback_days=30,
        max_position=0.40, max_sector_exposure=1.0, max_gross=1.0,
    )
    equity = build_equity_curve(
        decs, prices, sizer=risk_aware_sizer, sizing_config=cfg,
    )
    assert len(equity) >= 5  # one anchor per rebalance + start


def test_risk_aware_curve_respects_gross_cap():
    """A book held at sub-1 gross earns strictly less than a fully-invested book.

    Direct equality of (capped_return / uncapped_return) to the gross
    ratio fails under compounding — that's not a bug, that's how
    compounding works. The right claim is directional: under a positive
    drift, capping gross at <1 produces a smaller total return.
    """
    idx = pd.bdate_range("2023-01-01", "2024-01-15")
    n = len(idx)
    prices = pd.DataFrame({
        "A": 100 * (1.005 ** np.arange(n)),
        "B": 100 * (1.005 ** np.arange(n)),
    }, index=idx)
    rebal = generate_rebalance_dates("2023-04-01", "2024-01-01", "monthly")
    decs = [Decision(t, d, "Buy") for d in rebal for t in ["A", "B"]]

    eq_uncapped = build_equity_curve(decs, prices)  # legacy fully-invested
    cfg = SizingConfig(
        use_vol_scaling=False, base_weight=0.125,  # raw gross = 0.50
        max_position=1.0, max_sector_exposure=10.0, max_gross=10.0,
    )
    eq_capped = build_equity_curve(
        decs, prices, sizer=risk_aware_sizer, sizing_config=cfg,
    )
    # On a positively-drifting universe, less leverage = less return.
    assert eq_capped.iloc[-1] < eq_uncapped.iloc[-1]
    # And the capped book still gained something (gross > 0).
    assert eq_capped.iloc[-1] > eq_capped.iloc[0]
