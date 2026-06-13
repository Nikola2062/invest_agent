"""Tests for the Price Map Agent — deterministic fallback path + validation."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.price_map import (
    _deterministic_fallback, _validate_price_map, build_price_map,
)
from src.models.schemas import (
    CostBasisLevel, CostBasisMap, PriceMap, PriceMapZone,
    RelativeStrengthAssessment, StructureAssessment, StructurePivot,
    VolumeAssessment,
)


def _stub_inputs():
    structure = StructureAssessment(
        trend="downtrend", stage="middle", confidence=0.7,
        last_swing_high=35.0, last_swing_low=22.0,
        pivots=[StructurePivot(date="2025-04-01", price=35.0, kind="LH")],
        structure_summary="3 LL + 2 LH",
    )
    volume = VolumeAssessment(
        institutional_flow="distribution", obv_trend="falling",
        volume_expansion_pct=-10.0, up_down_volume_ratio=0.6,
        last_earnings_volume_spike_x=None, confidence=0.7, signals=["OBV falling"],
    )
    cost_basis = CostBasisMap(
        lookback_days=365,
        hvn_levels=[
            CostBasisLevel(price_low=20.0, price_high=22.0, volume_pct_of_window=15.0,
                           position_vs_current="below", role="support"),
            CostBasisLevel(price_low=30.0, price_high=32.0, volume_pct_of_window=20.0,
                           position_vs_current="above", role="resistance"),
        ],
        trapped_supply_pct=20.0, accumulation_pct=15.0,
        summary="HVN around 21 and 31",
    )
    rs = RelativeStrengthAssessment(
        vs_sector_etf_90d=0.7, vs_sector_etf_365d=0.5,
        vs_index_90d=0.6, vs_index_365d=0.4,
        rs_rank_in_universe=None, signal="laggard",
        benchmark_sector_etf="IGV", benchmark_index="SPY",
    )
    return structure, volume, cost_basis, rs


def test_deterministic_fallback_produces_valid_map():
    structure, volume, cost_basis, rs = _stub_inputs()
    pm = build_price_map(
        symbol="FIG", current_price=25.0, currency="USD",
        intrinsic_low=22.0, intrinsic_base=28.0, intrinsic_high=38.0,
        structure=structure, volume=volume, cost_basis=cost_basis, relative_strength=rs,
        enable_llm=False,
    )
    ok, msg = _validate_price_map(pm, current_price=25.0)
    assert ok, f"validation failed: {msg}"
    assert pm.key_support is not None
    assert pm.key_resistance is not None
    print(f"✓ Fallback price map valid; current zone = {pm.zones[pm.current_zone_index].label}, "
          f"key_support={pm.key_support}, key_resistance={pm.key_resistance}")


def test_validator_rejects_misordered_zones():
    bad = PriceMap(
        zones=[
            PriceMapZone(price_low=30.0, price_high=40.0, label="hold", rationale="x"),
            PriceMapZone(price_low=20.0, price_high=25.0, label="accumulation", rationale="x"),
        ],
        current_zone_index=0, summary="bad",
    )
    ok, msg = _validate_price_map(bad, current_price=35.0)
    assert not ok
    assert "sorted" in (msg or "")
    print(f"✓ Validator rejects misordered zones: {msg}")


def test_validator_rejects_bad_current_zone():
    pm = PriceMap(
        zones=[
            PriceMapZone(price_low=20.0, price_high=25.0, label="accumulation", rationale="x"),
            PriceMapZone(price_low=25.0, price_high=30.0, label="hold", rationale="x"),
        ],
        current_zone_index=0, summary="ok zones, wrong idx",
    )
    ok, msg = _validate_price_map(pm, current_price=28.0)
    assert not ok
    assert "not inside" in (msg or "")
    print(f"✓ Validator rejects wrong current_zone_index: {msg}")


def test_llm_failure_falls_back_to_deterministic():
    """When the LLM call itself raises, build_price_map must return the fallback."""
    from src.agents import price_map as pm_module
    original = pm_module.chat_json

    def _boom(**kwargs):
        raise RuntimeError("simulated LLM outage")
    pm_module.chat_json = _boom
    try:
        structure, volume, cost_basis, rs = _stub_inputs()
        pm = build_price_map(
            symbol="FIG", current_price=25.0, currency="USD",
            intrinsic_low=22.0, intrinsic_base=28.0, intrinsic_high=38.0,
            structure=structure, volume=volume, cost_basis=cost_basis, relative_strength=rs,
            enable_llm=True,
        )
    finally:
        pm_module.chat_json = original
    ok, msg = _validate_price_map(pm, current_price=25.0)
    assert ok, f"fallback invalid after LLM failure: {msg}"
    assert "fallback" in pm.summary.lower()
    print("✓ LLM outage → deterministic fallback returned")
