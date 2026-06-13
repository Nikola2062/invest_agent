"""Tests for the Cost-Basis Agent — volume-by-price histogram."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.agents.cost_basis import assess_cost_basis
from src.models.schemas import CostBasisCfg


def _bars(closes: list[float], volumes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({
        "Open": closes,
        "High": [c * 1.005 for c in closes],
        "Low":  [c * 0.995 for c in closes],
        "Close": closes,
        "Volume": volumes,
    }, index=dates)


def test_hvn_detected_at_concentrated_zone():
    # 200 bars: 150 of them clustered around $50, 50 around $30
    closes = [50.0] * 150 + [30.0] * 50
    volumes = [1_000_000] * 200
    out = assess_cost_basis(_bars(closes, volumes), CostBasisCfg(), current_price=40.0)
    # At least one HVN should land near $50 (the heavy cluster)
    near_50 = [lv for lv in out.hvn_levels if lv.price_low <= 50.0 <= lv.price_high]
    assert near_50, f"expected HVN around $50; got levels {[(lv.price_low, lv.price_high) for lv in out.hvn_levels]}"
    # Since current_price=$40, the $50 cluster sits ABOVE → resistance/trapped supply
    above_levels = [lv for lv in out.hvn_levels if lv.position_vs_current == "above"]
    assert above_levels, "expected at least one HVN above current"
    assert all(lv.role == "resistance" for lv in above_levels)
    print(f"✓ Heavy cluster at $50 detected as resistance from current $40 (trapped_supply {out.trapped_supply_pct}%)")


def test_accumulation_zone_below_current():
    # Heavy volume at $45-46 (just below current $50)
    closes = [45.5] * 200 + [50.0] * 20
    volumes = [2_000_000] * 200 + [500_000] * 20
    out = assess_cost_basis(_bars(closes, volumes), CostBasisCfg(), current_price=50.0)
    assert out.accumulation_pct > 50.0, f"expected high accumulation; got {out.accumulation_pct}%"
    below = [lv for lv in out.hvn_levels if lv.position_vs_current == "below"]
    assert below, "expected an HVN below current as a support level"
    assert all(lv.role == "support" for lv in below)
    print(f"✓ Heavy cluster below current → support level + accumulation_pct={out.accumulation_pct}%")


def test_insufficient_history():
    out = assess_cost_basis(_bars([50.0] * 10, [1_000_000] * 10), CostBasisCfg(), current_price=50.0)
    assert out.summary == "insufficient_history"
    assert out.hvn_levels == []
    print("✓ <30 bars → insufficient_history")
