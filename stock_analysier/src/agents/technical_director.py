"""Technical Director — aggregator that produces the composite signal from the
five Technical Division sub-outputs via a deterministic vote.

Voting scheme (per agent, signed integer in [-2, +2]):
  Structure:
    strong_uptrend → +2, uptrend → +1, range → 0, downtrend → -1, strong_downtrend → -2
  Volume:
    accumulation → +1, neutral → 0, distribution → -1
    (rising OBV adds +1, falling OBV adds -1, capped at ±2)
  Cost-basis:
    accumulation_pct - trapped_supply_pct > +20  → +1
    accumulation_pct - trapped_supply_pct < -20  → -1
    else 0
  Relative strength:
    strong_leader → +2, leader → +1, neutral → 0, laggard → -1, weak_laggard → -2

Composite = sum across 4 sub-signals (range ≈ [-7, +7]). Banded to the 5-level signal.

This is deliberately deterministic — the LLM is the Price Map narrator, not the
arbiter of the composite signal.
"""
from __future__ import annotations

from src.models.schemas import (
    CostBasisMap,
    PriceMap,
    RelativeStrengthAssessment,
    StructureAssessment,
    TechnicalAssessment,
    VolumeAssessment,
)


def _structure_vote(s: StructureAssessment) -> int:
    return {
        "strong_uptrend": 2, "uptrend": 1, "range": 0,
        "downtrend": -1, "strong_downtrend": -2,
    }.get(s.trend, 0)


def _volume_vote(v: VolumeAssessment) -> int:
    flow_pts = {"accumulation": 1, "neutral": 0, "distribution": -1}.get(v.institutional_flow, 0)
    obv_pts = {"rising": 1, "flat": 0, "falling": -1}.get(v.obv_trend, 0)
    raw = flow_pts + obv_pts
    return max(-2, min(2, raw))


def _cost_basis_vote(cb: CostBasisMap) -> int:
    delta = cb.accumulation_pct - cb.trapped_supply_pct
    if delta > 20:
        return 1
    if delta < -20:
        return -1
    return 0


def _rs_vote(rs: RelativeStrengthAssessment) -> int:
    return {
        "strong_leader": 2, "leader": 1, "neutral": 0,
        "laggard": -1, "weak_laggard": -2,
    }.get(rs.signal, 0)


def composite_signal(
    structure: StructureAssessment,
    volume: VolumeAssessment,
    cost_basis: CostBasisMap,
    relative_strength: RelativeStrengthAssessment,
) -> tuple[str, str]:
    """Return (composite_signal_label, rationale_string)."""
    s = _structure_vote(structure)
    v = _volume_vote(volume)
    c = _cost_basis_vote(cost_basis)
    r = _rs_vote(relative_strength)
    total = s + v + c + r

    if total >= 5:
        label = "strong_bullish"
    elif total >= 2:
        label = "bullish"
    elif total >= -1:
        label = "neutral"
    elif total >= -4:
        label = "bearish"
    else:
        label = "strong_bearish"

    rationale = (
        f"Composite vote {total:+d} (structure {s:+d} [{structure.trend}], "
        f"volume {v:+d} [{volume.institutional_flow}/{volume.obv_trend}], "
        f"cost-basis {c:+d} (acc {cost_basis.accumulation_pct:.0f}% vs trapped {cost_basis.trapped_supply_pct:.0f}%), "
        f"RS {r:+d} [{relative_strength.signal}])."
    )
    return label, rationale


def assemble(
    structure: StructureAssessment,
    volume: VolumeAssessment,
    cost_basis: CostBasisMap,
    relative_strength: RelativeStrengthAssessment,
    price_map: PriceMap,
) -> TechnicalAssessment:
    label, rationale = composite_signal(structure, volume, cost_basis, relative_strength)
    return TechnicalAssessment(
        structure=structure,
        volume=volume,
        cost_basis=cost_basis,
        relative_strength=relative_strength,
        price_map=price_map,
        composite_signal=label,
        composite_rationale=rationale,
    )
