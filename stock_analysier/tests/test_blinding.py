"""Architectural test: the deterministic Technical Division agents must NOT
accept any fundamental field via their function signatures.

This is enforced by inspecting each agent's signature and asserting the allowed
parameter names. If someone adds a `fundamental=` kwarg later, this test fails.
"""
from __future__ import annotations

import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.cost_basis import assess_cost_basis
from src.agents.relative_strength import assess_relative_strength
from src.agents.structure import assess_structure
from src.agents.technical_director import assemble, composite_signal
from src.agents.volume_flow import assess_volume

# Fundamental-side names that must NEVER appear in a deterministic chart agent's signature.
_FORBIDDEN = {
    "fundamental",
    "fundamentals",
    "valuation",
    "risk",
    "macro",
    "forward_catalysts",
    "devil_advocate",
    "quality_score",
    "moat_strength",
    "margin_of_safety_pct",
    "intrinsic_low",
    "intrinsic_base",
    "intrinsic_high",
    "thesis",
    "thesis_one_liner",
    "recommendation",
    "red_flags",
    "balance_sheet_health",
}


def _params(fn) -> set[str]:
    return set(inspect.signature(fn).parameters.keys())


def test_structure_agent_blinded():
    forbidden_in_use = _params(assess_structure) & _FORBIDDEN
    assert not forbidden_in_use, f"Structure Agent leaks fundamental params: {forbidden_in_use}"


def test_volume_agent_blinded():
    forbidden_in_use = _params(assess_volume) & _FORBIDDEN
    assert not forbidden_in_use, f"Volume Agent leaks fundamental params: {forbidden_in_use}"


def test_cost_basis_agent_blinded():
    """Cost-Basis needs `current_price` which is a pure number — not fundamental output."""
    forbidden_in_use = _params(assess_cost_basis) & _FORBIDDEN
    assert not forbidden_in_use, f"Cost-Basis Agent leaks fundamental params: {forbidden_in_use}"


def test_relative_strength_agent_blinded():
    forbidden_in_use = _params(assess_relative_strength) & _FORBIDDEN
    assert not forbidden_in_use, f"Relative Strength Agent leaks fundamental params: {forbidden_in_use}"


def test_technical_director_blinded():
    # Aggregator only consumes the 5 technical sub-assessments + price map; no fundamentals.
    forbidden_in_use = _params(assemble) & _FORBIDDEN
    forbidden_in_use |= _params(composite_signal) & _FORBIDDEN
    assert not forbidden_in_use, f"Technical Director leaks fundamental params: {forbidden_in_use}"


def test_composite_signal_smoke():
    """The vote computation itself: feed synthetic outputs, verify it bands correctly."""
    from src.models.schemas import (
        CostBasisLevel, CostBasisMap, RelativeStrengthAssessment,
        StructureAssessment, StructurePivot, VolumeAssessment,
    )
    # All four agents firing bullish at maximum
    s = StructureAssessment(
        trend="strong_uptrend", stage="middle", confidence=0.9,
        last_swing_high=100.0, last_swing_low=70.0,
        pivots=[StructurePivot(date="2025-01-01", price=100.0, kind="HH")],
        structure_summary="all HH/HL",
    )
    v = VolumeAssessment(
        institutional_flow="accumulation", obv_trend="rising",
        volume_expansion_pct=25.0, up_down_volume_ratio=2.0,
        last_earnings_volume_spike_x=4.0, confidence=0.9,
        signals=["OBV rising"],
    )
    cb = CostBasisMap(
        lookback_days=365,
        hvn_levels=[CostBasisLevel(price_low=80, price_high=82, volume_pct_of_window=30,
                                   position_vs_current="below", role="support")],
        trapped_supply_pct=5.0, accumulation_pct=40.0,
        summary="strong accumulation",
    )
    rs = RelativeStrengthAssessment(
        vs_sector_etf_90d=1.5, vs_sector_etf_365d=2.0,
        vs_index_90d=1.4, vs_index_365d=1.9,
        rs_rank_in_universe=None, signal="strong_leader",
        benchmark_sector_etf="IGV", benchmark_index="SPY",
    )
    label, rationale = composite_signal(s, v, cb, rs)
    assert label == "strong_bullish", f"got {label}; rationale: {rationale}"
    print(f"✓ All-bullish stub → {label}")

    # Flip everything bearish
    s2 = s.model_copy(update={"trend": "strong_downtrend"})
    v2 = v.model_copy(update={"institutional_flow": "distribution", "obv_trend": "falling"})
    cb2 = cb.model_copy(update={"trapped_supply_pct": 40.0, "accumulation_pct": 5.0})
    rs2 = rs.model_copy(update={"signal": "weak_laggard"})
    label2, _ = composite_signal(s2, v2, cb2, rs2)
    assert label2 == "strong_bearish", f"got {label2}"
    print(f"✓ All-bearish stub → {label2}")
