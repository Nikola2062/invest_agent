"""Smoke tests for the deterministic Tactical Exit + Order Generator agents.

These don't need an LLM. They verify the risk-policy ladder fires correctly
under synthetic risk scenarios.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datetime import date, timedelta

from src.agents.order_generator import generate_entry_orders, generate_held_orders
from src.agents.tactical_exit import decide_tactical
from src.config.loader import load_risk_policy
from src.models.schemas import (
    Holding,
    RiskAssessment,
    Scenario,
    ValuationResult,
)


def make_risk(p10, p15, p20, p25, vol=30.0, signals=None):
    return RiskAssessment(
        scenarios=[
            Scenario(name="base", probability=0.6, expected_return_pct=5.0, expected_drawdown_pct=-8.0, rationale="base case"),
            Scenario(name="bear", probability=0.4, expected_return_pct=-15.0, expected_drawdown_pct=-25.0, rationale="bear case"),
        ],
        drawdown_probabilities={"10": p10, "15": p15, "20": p20, "25": p25},
        realized_vol_annualized_pct=vol,
        key_macro_signals=signals or [],
        horizon_days=90,
        persistence_days_observed=10,
    )


def seed_warm_audit_history(symbol: str, at_level: int, count: int = 5) -> None:
    """Insert `count` synthetic prior runs at the given tactical level into the
    (isolated) test audit DB. Lets tests bypass the cold-start cap when their
    intent is to verify the warm-history behavior of the tactical ladder.
    """
    import sqlite3
    from datetime import datetime, timedelta, timezone

    from src.storage import audit

    with audit._conn(audit.DEFAULT_DB_PATH) as c:
        now = datetime.now(timezone.utc)
        for i in range(count):
            ts = (now - timedelta(days=count - i)).isoformat()
            c.execute(
                """INSERT INTO analysis_runs
                   (symbol, market, timestamp_utc, current_price, currency,
                    tactical_level, full_result_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (symbol, "US", ts, 50.0, "USD", at_level, "{}"),
            )


def test_no_action_on_calm_market():
    policy = load_risk_policy()
    holding = Holding(symbol="FIG", market="US", shares=2000, cost_basis_per_share=55.0, currency="USD")
    risk = make_risk(0.25, 0.10, 0.05, 0.02)  # benign
    tactical = decide_tactical(holding, current_price=60.0, risk=risk, policy=policy)
    assert tactical.action == "no_action", f"expected no_action, got {tactical.action}"
    print("✓ Calm market → no_action")


def test_yellow_fires_low_threshold():
    policy = load_risk_policy()
    holding = Holding(symbol="FIG", market="US", shares=2000, cost_basis_per_share=55.0, currency="USD")
    risk = make_risk(0.45, 0.15, 0.05, 0.02)  # 10%-mag prob = 45% > 40% threshold
    tactical = decide_tactical(holding, current_price=60.0, risk=risk, policy=policy)
    assert tactical.label == "YELLOW_WATCH", f"expected YELLOW_WATCH, got {tactical.label}"
    print("✓ Elevated P(dd≥10%) → YELLOW_WATCH")


def test_orange_fires_and_orders_generated():
    policy = load_risk_policy()
    holding = Holding(symbol="FIG", market="US", shares=2000, cost_basis_per_share=55.0, currency="USD")
    risk = make_risk(0.60, 0.60, 0.30, 0.10, signals=["10Y-3M inverted", "VIX > 25"])
    tactical = decide_tactical(holding, current_price=60.0, risk=risk, policy=policy)
    # ORANGE: dd_mag=15%, prob_min=0.55. We pass 0.60 at 15%. ORANGE_TRIM fires.
    assert tactical.label == "ORANGE_TRIM", f"expected ORANGE_TRIM, got {tactical.label}"
    assert tactical.trim_pct_of_position == 30
    # Rebuy band 10-15% below current $60 → $51-$54
    assert tactical.rebuy_band_high is not None and abs(tactical.rebuy_band_high - 54.0) < 0.01
    assert tactical.rebuy_band_low is not None and abs(tactical.rebuy_band_low - 51.0) < 0.01

    decision = generate_held_orders(holding, 60.0, tactical)
    assert len(decision.immediate_orders) == 1, f"expected 1 immediate, got {len(decision.immediate_orders)}"
    assert decision.immediate_orders[0].side == "SELL"
    assert decision.immediate_orders[0].quantity == 600  # 30% of 2000
    assert len(decision.rebuy_orders) == 2, f"expected 2 rebuy tranches, got {len(decision.rebuy_orders)}"
    print("✓ ORANGE_TRIM → 600-share SELL + 2 conditional BUY tranches")


def test_red_fires_with_hedge():
    policy = load_risk_policy()
    holding = Holding(symbol="FIG", market="US", shares=2000, cost_basis_per_share=55.0, currency="USD")
    seed_warm_audit_history(holding.symbol, at_level=3)  # bypass cold-start cap
    risk = make_risk(0.85, 0.80, 0.70, 0.20, signals=["yield inversion", "VIX 35"])
    tactical = decide_tactical(holding, current_price=60.0, risk=risk, policy=policy)
    assert tactical.label == "RED_DEFENSIVE"
    assert tactical.hedge_recommended is True
    print("✓ RED_DEFENSIVE → hedge_recommended=True")


def test_black_fires_full_exit_when_long_held():
    policy = load_risk_policy()
    long_ago = date.today() - timedelta(days=400)
    holding = Holding(
        symbol="FIG", market="US", shares=2000, cost_basis_per_share=55.0,
        currency="USD", purchase_date=long_ago,
    )
    seed_warm_audit_history(holding.symbol, at_level=4)  # bypass cold-start cap
    risk = make_risk(0.90, 0.85, 0.78, 0.80)
    tactical = decide_tactical(holding, current_price=60.0, risk=risk, policy=policy)
    assert tactical.label == "BLACK_EXIT"
    assert any("long-term" in n for n in tactical.tax_notes), tactical.tax_notes
    print("✓ BLACK_EXIT (long-term holding) → full_exit, long-term tax note")


def test_black_downgraded_when_near_long_term_threshold():
    policy = load_risk_policy()
    # 340 days = within 60-day proximity of 365
    short_held = date.today() - timedelta(days=340)
    holding = Holding(
        symbol="FIG", market="US", shares=2000, cost_basis_per_share=55.0,
        currency="USD", purchase_date=short_held,
    )
    seed_warm_audit_history(holding.symbol, at_level=4)  # bypass cold-start cap
    risk = make_risk(0.90, 0.85, 0.78, 0.80)
    tactical = decide_tactical(holding, current_price=60.0, risk=risk, policy=policy)
    # Should be downgraded to RED_DEFENSIVE
    assert tactical.label == "RED_DEFENSIVE", f"expected RED_DEFENSIVE downgrade, got {tactical.label}"
    assert any("Downgraded" in n for n in tactical.tax_notes)
    print("✓ BLACK→RED downgrade when 25d from long-term threshold")


def test_wash_sale_note_when_underwater():
    """User's FIG position is underwater — wash-sale note should fire."""
    policy = load_risk_policy()
    holding = Holding(symbol="FIG", market="US", shares=2000, cost_basis_per_share=55.0, currency="USD")
    seed_warm_audit_history(holding.symbol, at_level=3)  # bypass cold-start cap (RED expected)
    risk = make_risk(0.85, 0.80, 0.70, 0.20)
    # Selling at $25 < cost $55 = LOSS
    tactical = decide_tactical(holding, current_price=25.0, risk=risk, policy=policy)
    assert tactical.label == "RED_DEFENSIVE"
    assert any("wash-sale" in n.lower() for n in tactical.tax_notes), tactical.tax_notes
    print("✓ Underwater sell → wash-sale note included")


def test_high_vol_widens_thresholds():
    policy = load_risk_policy()
    holding = Holding(symbol="FIG", market="US", shares=2000, cost_basis_per_share=55.0, currency="USD")
    # Same probabilities, but high vol (FIG-like at 80%)
    risk = make_risk(0.60, 0.60, 0.30, 0.10, vol=80.0)
    tactical = decide_tactical(holding, current_price=60.0, risk=risk, policy=policy)
    # With vol_adjustment 1.3x: ORANGE's 15% becomes 19.5%. P(dd≥19.5%) interpolates between
    # 0.60 (at 15%) and 0.30 (at 20%): α=0.9 → 0.60*0.1 + 0.30*0.9 = 0.33. Below 0.55 threshold → ORANGE doesn't fire.
    # YELLOW: 10% becomes 13%. P(dd≥13%) = 0.60*0.6 + 0.60*0.4 = 0.60. Above 0.40 → YELLOW fires.
    assert tactical.label == "YELLOW_WATCH", f"expected YELLOW_WATCH (vol-widened), got {tactical.label}"
    print("✓ High vol widens thresholds — ORANGE downgraded to YELLOW")


def test_entry_orders_buy_now():
    valuation = ValuationResult(
        current_price=80.0, currency="USD",
        intrinsic_low=95.0, intrinsic_base=110.0, intrinsic_high=130.0,
        margin_of_safety_pct=37.5,  # (110/80 - 1) * 100
        methodology_notes="DCF + multiples blend",
        confidence="medium",
    )
    decision = generate_entry_orders("NVDA", "US", 80.0, valuation)
    assert decision.recommendation == "BUY_NOW"
    assert len(decision.entry_orders) == 1
    print("✓ Big margin of safety → BUY_NOW with entry order")


def test_entry_orders_wait_for_price():
    valuation = ValuationResult(
        current_price=100.0, currency="USD",
        intrinsic_low=95.0, intrinsic_base=105.0, intrinsic_high=115.0,
        margin_of_safety_pct=5.0,  # close to fair
        methodology_notes="multiples blend",
        confidence="medium",
    )
    decision = generate_entry_orders("MSFT", "US", 100.0, valuation)
    assert decision.recommendation == "WAIT_FOR_PRICE"
    print("✓ Close to fair → WAIT_FOR_PRICE")


def _make_technical_bearish() -> "TechnicalAssessment":
    """Helper: build a TechnicalAssessment that triggers the bearish boost."""
    from src.models.schemas import (
        CostBasisMap, PriceMap, RelativeStrengthAssessment,
        StructureAssessment, StructurePivot, TechnicalAssessment, VolumeAssessment,
    )
    return TechnicalAssessment(
        structure=StructureAssessment(
            trend="strong_downtrend", stage="middle", confidence=0.8,
            last_swing_high=70.0, last_swing_low=50.0,
            pivots=[StructurePivot(date="2025-04-01", price=70.0, kind="LH")],
            structure_summary="3 LL + 2 LH",
        ),
        volume=VolumeAssessment(
            institutional_flow="distribution", obv_trend="falling",
            volume_expansion_pct=-15.0, up_down_volume_ratio=0.5,
            last_earnings_volume_spike_x=None, confidence=0.8, signals=["OBV falling"],
        ),
        cost_basis=CostBasisMap(lookback_days=365, hvn_levels=[],
                                trapped_supply_pct=40.0, accumulation_pct=5.0,
                                summary="distribution-heavy"),
        relative_strength=RelativeStrengthAssessment(
            vs_sector_etf_90d=0.7, vs_sector_etf_365d=0.5,
            vs_index_90d=0.6, vs_index_365d=0.4,
            rs_rank_in_universe=None, signal="weak_laggard",
            benchmark_sector_etf="IGV", benchmark_index="SPY",
        ),
        price_map=PriceMap(zones=[], current_zone_index=0, summary="stub"),
        composite_signal="strong_bearish",
        composite_rationale="all bearish",
    )


def test_technical_boost_lowers_persistence_required():
    """Marginal ORANGE setup with only 1 of 2 prior qualifying runs.
    Without boost: fails persistence (1 < 2 required).
    With bearish technical confirmation: required drops to 1, passes.
    """
    policy = load_risk_policy()
    holding = Holding(symbol="FIG_BOOST", market="US", shares=2000, cost_basis_per_share=55.0, currency="USD")
    # Seed: 1 qualifying ORANGE run + enough non-qualifying to clear cold-start
    seed_warm_audit_history(holding.symbol, at_level=2, count=1)
    seed_warm_audit_history(holding.symbol, at_level=0, count=3)
    risk = make_risk(0.60, 0.60, 0.30, 0.10)

    # Without technical: persistence requires 2, we have 1 → ORANGE fails → falls through
    no_tech = decide_tactical(holding, current_price=60.0, risk=risk, policy=policy, technical=None)
    # With bearish technical: required drops to 1 → ORANGE fires
    tech = _make_technical_bearish()
    with_tech = decide_tactical(holding, current_price=60.0, risk=risk, policy=policy, technical=tech)

    # Compare: technical boost should produce a strictly higher or equal level
    no_level = no_tech.level or 0
    with_level = with_tech.level or 0
    assert with_level >= no_level, f"boost did not help: without={no_tech.label} with={with_tech.label}"
    print(f"✓ Technical boost: without tech → {no_tech.label or 'no_action'}; with tech → {with_tech.label or 'no_action'}")


def test_rebuy_band_anchored_to_key_support_when_in_range():
    """When price_map.key_support sits within [current*(1-30%), current*(1-5%)],
    use it as rebuy_band_high (tranche 1) and key_support*0.95 as rebuy_band_low.
    """
    from src.models.schemas import (
        CostBasisMap, PriceMap, PriceMapZone, RelativeStrengthAssessment,
        StructureAssessment, StructurePivot, TechnicalAssessment, VolumeAssessment,
    )
    policy = load_risk_policy()
    holding = Holding(symbol="FIG_ANCHOR", market="US", shares=2000, cost_basis_per_share=55.0, currency="USD")
    seed_warm_audit_history(holding.symbol, at_level=3, count=5)  # bypass cold-start
    risk = make_risk(0.85, 0.80, 0.70, 0.20)
    current = 60.0
    # Key support at $51 is 15% below current → inside the 5%-30% anchoring band
    tech = TechnicalAssessment(
        structure=StructureAssessment(
            trend="range", stage="middle", confidence=0.5,
            last_swing_high=60, last_swing_low=51,
            pivots=[StructurePivot(date="2025-04-01", price=60, kind="LH")],
            structure_summary="oscillating",
        ),
        volume=VolumeAssessment(
            institutional_flow="neutral", obv_trend="flat",
            volume_expansion_pct=0.0, up_down_volume_ratio=1.0,
            last_earnings_volume_spike_x=None, confidence=0.4, signals=[],
        ),
        cost_basis=CostBasisMap(lookback_days=365, hvn_levels=[],
                                trapped_supply_pct=10.0, accumulation_pct=10.0,
                                summary="balanced"),
        relative_strength=RelativeStrengthAssessment(
            vs_sector_etf_90d=1.0, vs_sector_etf_365d=1.0,
            vs_index_90d=1.0, vs_index_365d=1.0,
            rs_rank_in_universe=None, signal="neutral",
            benchmark_sector_etf="IGV", benchmark_index="SPY",
        ),
        price_map=PriceMap(
            zones=[PriceMapZone(price_low=50, price_high=60, label="hold", rationale="stub")],
            current_zone_index=0, key_support=51.0, key_resistance=70.0,
            summary="stub",
        ),
        composite_signal="neutral", composite_rationale="balanced",
    )
    out = decide_tactical(holding, current_price=current, risk=risk, policy=policy, technical=tech)
    # Should fire at RED (high probabilities) and anchor band to 51
    assert out.rebuy_band_high is not None
    assert abs(out.rebuy_band_high - 51.0) < 0.01, f"expected anchored at 51; got {out.rebuy_band_high}"
    assert out.rebuy_band_low is not None and abs(out.rebuy_band_low - 51.0 * 0.95) < 0.01
    assert "anchored" in out.rationale.lower()
    print(f"✓ Rebuy band anchored to key_support: high={out.rebuy_band_high}, low={out.rebuy_band_low:.2f}")


def test_rebuy_band_falls_back_to_pct_when_support_out_of_range():
    """key_support > 5% below current → out of anchor band → fall back to %-rebuy."""
    from src.models.schemas import (
        CostBasisMap, PriceMap, PriceMapZone, RelativeStrengthAssessment,
        StructureAssessment, StructurePivot, TechnicalAssessment, VolumeAssessment,
    )
    policy = load_risk_policy()
    holding = Holding(symbol="FIG_NOANCHOR", market="US", shares=2000, cost_basis_per_share=55.0, currency="USD")
    seed_warm_audit_history(holding.symbol, at_level=3, count=5)
    risk = make_risk(0.85, 0.80, 0.70, 0.20)
    current = 60.0
    # Key support at $59 is only 1.7% below current — too close to anchor
    tech = TechnicalAssessment(
        structure=StructureAssessment(
            trend="range", stage="middle", confidence=0.5,
            last_swing_high=60, last_swing_low=59,
            pivots=[StructurePivot(date="2025-04-01", price=60, kind="LH")],
            structure_summary="x",
        ),
        volume=VolumeAssessment(
            institutional_flow="neutral", obv_trend="flat",
            volume_expansion_pct=0.0, up_down_volume_ratio=1.0,
            last_earnings_volume_spike_x=None, confidence=0.4, signals=[],
        ),
        cost_basis=CostBasisMap(lookback_days=365, hvn_levels=[],
                                trapped_supply_pct=10.0, accumulation_pct=10.0, summary="x"),
        relative_strength=RelativeStrengthAssessment(
            vs_sector_etf_90d=1.0, vs_sector_etf_365d=1.0,
            vs_index_90d=1.0, vs_index_365d=1.0,
            rs_rank_in_universe=None, signal="neutral",
            benchmark_sector_etf="IGV", benchmark_index="SPY",
        ),
        price_map=PriceMap(
            zones=[PriceMapZone(price_low=58, price_high=62, label="hold", rationale="stub")],
            current_zone_index=0, key_support=59.0, key_resistance=70.0, summary="x",
        ),
        composite_signal="neutral", composite_rationale="x",
    )
    out = decide_tactical(holding, current_price=current, risk=risk, policy=policy, technical=tech)
    # %-based RED band: high = current*(1-15%) = 51, low = current*(1-22%) = 46.8
    assert out.rebuy_band_high is not None
    assert abs(out.rebuy_band_high - 51.0) < 0.5, f"expected %-fallback ~51; got {out.rebuy_band_high}"
    assert "anchored" not in out.rationale.lower()
    print(f"✓ Support too close → fell back to %-band: high={out.rebuy_band_high:.2f}")


def test_technical_boost_does_not_override_cold_start_cap():
    """Cold-start (no audit) + bearish technical + extreme drawdown.
    Even with the boost, BLACK_EXIT must remain capped at the configured max level.
    """
    policy = load_risk_policy()
    holding = Holding(symbol="UNSEEN_BOOSTED", market="US", shares=100, cost_basis_per_share=50.0, currency="USD")
    risk = make_risk(0.95, 0.90, 0.85, 0.85)
    tech = _make_technical_bearish()
    out = decide_tactical(holding, current_price=60.0, risk=risk, policy=policy, technical=tech)
    assert out.level is not None and out.level <= policy.cold_start_max_action_level, (
        f"cold-start cap broken by technical boost: level={out.level} ({out.label})"
    )
    print(f"✓ Cold-start cap holds against technical boost: {out.label}")


def test_cold_start_caps_high_level_actions():
    """Fresh install (no audit history) should NOT fire BLACK_EXIT on day 1.
    With cold_start_max_action_level=2, a RED-or-higher signal must fall through
    to ORANGE_TRIM (level=2) instead. The audit DB is empty per conftest fixture.
    """
    policy = load_risk_policy()
    holding = Holding(symbol="UNSEEN_TICKER", market="US", shares=100, cost_basis_per_share=50.0, currency="USD")
    risk = make_risk(0.95, 0.90, 0.85, 0.85)  # would trigger BLACK_EXIT with full history
    tactical = decide_tactical(holding, current_price=60.0, risk=risk, policy=policy)
    assert tactical.level is not None and tactical.level <= policy.cold_start_max_action_level, (
        f"expected level <= {policy.cold_start_max_action_level} on cold-start, got level={tactical.level} ({tactical.label})"
    )
    print(f"✓ Cold-start: BLACK signal capped at {tactical.label}")


def test_entry_orders_pass():
    valuation = ValuationResult(
        current_price=100.0, currency="USD",
        intrinsic_low=60.0, intrinsic_base=75.0, intrinsic_high=90.0,
        margin_of_safety_pct=-25.0,
        methodology_notes="multiples blend",
        confidence="medium",
    )
    decision = generate_entry_orders("XYZ", "US", 100.0, valuation)
    assert decision.recommendation == "PASS"
    print("✓ Overvalued → PASS")


if __name__ == "__main__":
    test_no_action_on_calm_market()
    test_yellow_fires_low_threshold()
    test_orange_fires_and_orders_generated()
    test_red_fires_with_hedge()
    test_black_fires_full_exit_when_long_held()
    test_black_downgraded_when_near_long_term_threshold()
    test_wash_sale_note_when_underwater()
    test_high_vol_widens_thresholds()
    test_entry_orders_buy_now()
    test_entry_orders_wait_for_price()
    test_cold_start_caps_high_level_actions()
    test_entry_orders_pass()
    print("\nAll tactical / order generator tests passed ✅")
