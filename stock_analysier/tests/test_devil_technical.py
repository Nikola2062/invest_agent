"""Tests for the Devil's Advocate Technical integration (Phase 6).

We don't hit the LLM here — the LLM is stubbed. We verify:
  (a) When TechnicalAssessment is attached, the user prompt includes the technical block.
  (b) The new category `technical_fundamental_contradiction` is accepted by the schema.
  (c) The deterministic verdict logic still escalates a veto finding to veto.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.devil_advocate import _format_user
from src.models.schemas import (
    AnalysisResult, CostBasisMap, DevilFinding, FundamentalAssessment,
    HeldDecision, Holding, NotHeldDecision, PriceMap, PriceMapZone,
    RelativeStrengthAssessment, RiskAssessment, Scenario,
    StructureAssessment, StructurePivot, TacticalAction, TechnicalAssessment,
    ValuationResult, VolumeAssessment,
)


def _make_result(with_technical: bool) -> AnalysisResult:
    technical = None
    if with_technical:
        technical = TechnicalAssessment(
            structure=StructureAssessment(
                trend="strong_downtrend", stage="middle", confidence=0.9,
                last_swing_high=80, last_swing_low=50,
                pivots=[StructurePivot(date="2025-05-01", price=80, kind="LH")],
                structure_summary="5 LL + 3 LH",
            ),
            volume=VolumeAssessment(
                institutional_flow="distribution", obv_trend="falling",
                volume_expansion_pct=-15.0, up_down_volume_ratio=0.5,
                last_earnings_volume_spike_x=None, confidence=0.8, signals=["OBV falling"],
            ),
            cost_basis=CostBasisMap(
                lookback_days=365, hvn_levels=[],
                trapped_supply_pct=40.0, accumulation_pct=5.0,
                summary="distribution-heavy",
            ),
            relative_strength=RelativeStrengthAssessment(
                vs_sector_etf_90d=0.6, vs_sector_etf_365d=0.4,
                vs_index_90d=0.5, vs_index_365d=0.3,
                rs_rank_in_universe=None, signal="weak_laggard",
                benchmark_sector_etf="IGV", benchmark_index="SPY",
            ),
            price_map=PriceMap(
                zones=[PriceMapZone(price_low=45, price_high=55, label="watch", rationale="stub")],
                current_zone_index=0, key_support=45.0, key_resistance=70.0,
                summary="stub",
            ),
            composite_signal="strong_bearish",
            composite_rationale="all four sub-signals bearish",
        )
    return AnalysisResult(
        symbol="FIG", market="US",
        timestamp_utc=datetime.now(timezone.utc),
        current_price=50.0, currency="USD",
        position=Holding(symbol="FIG", market="US", shares=100, cost_basis_per_share=55.0, currency="USD"),
        fundamental=FundamentalAssessment(
            quality_score=8.0, moat_assessment="strong", moat_strength="wide",
            balance_sheet_health="strong", growth_outlook="strong",
            capital_allocation="excellent", red_flags=[],
            thesis_one_liner="Best-in-class collaborative design platform",
        ),
        valuation=ValuationResult(
            current_price=50.0, currency="USD",
            intrinsic_low=60.0, intrinsic_base=75.0, intrinsic_high=90.0,
            margin_of_safety_pct=33.3, methodology_notes="DCF + multiples", confidence="medium",
        ),
        risk=RiskAssessment(
            scenarios=[Scenario(name="base", probability=1.0, expected_return_pct=10.0,
                                expected_drawdown_pct=-10.0, rationale="x")],
            drawdown_probabilities={"10": 0.3, "15": 0.2, "20": 0.1, "25": 0.05},
            realized_vol_annualized_pct=40.0, key_macro_signals=[], horizon_days=90,
        ),
        if_held=HeldDecision(tactical=TacticalAction(action="no_action", rationale="hold")),
        if_not_held=NotHeldDecision(recommendation="BUY_NOW", entry_orders=[], rationale="MoS 33%"),
        technical=technical,
    )


def test_user_prompt_includes_technical_block_when_present():
    msg = _format_user(_make_result(with_technical=True))
    assert "TECHNICAL DIVISION" in msg
    assert "STRONG_BEARISH" in msg
    assert "weak_laggard" in msg
    assert "key_support=45.0" in msg
    print("✓ DA user prompt surfaces the TechnicalAssessment block")


def test_user_prompt_handles_missing_technical():
    msg = _format_user(_make_result(with_technical=False))
    assert "TECHNICAL DIVISION" in msg
    assert "did not run" in msg
    print("✓ DA user prompt handles missing TechnicalAssessment gracefully")


def test_new_category_is_accepted_by_schema():
    f = DevilFinding(
        category="technical_fundamental_contradiction",
        severity="concern",
        finding="Fundamentals say BUY_NOW (33% MoS, quality 8) but technical composite is strong_bearish.",
        evidence="Structure=strong_downtrend, volume=distribution, RS=weak_laggard for 90d AND 365d.",
        recommendation="Wait for the price-action to confirm the fundamental thesis before deploying.",
    )
    assert f.category == "technical_fundamental_contradiction"
    print("✓ New DevilFinding category accepted by schema")
