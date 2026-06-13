"""Price Map Agent — small LLM synthesizer that converts the 4 deterministic
technical outputs + the raw intrinsic-range price points into a zone→action map.

BLINDING: receives prices, the 4 technical assessments, and only the *numeric*
intrinsic_low/base/high from valuation. Does NOT receive the LLM-derived BUY/SELL
recommendation, MoS verdict, fundamental quality score, or any other narrative
field that would let the model rationalize after-the-fact.

Includes a deterministic fallback that builds a Price Map from the cost-basis
HVN levels alone, used when the LLM call is disabled or fails validation twice.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.llm.client import chat_json
from src.models.schemas import (
    CostBasisMap,
    PriceMap,
    PriceMapZone,
    RelativeStrengthAssessment,
    StructureAssessment,
    VolumeAssessment,
)

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a price-action strategist building a zone-by-zone action map for a stock.

You receive ONLY:
  - Current price
  - Structure verdict (trend / stage)
  - Volume verdict (institutional flow / OBV trend)
  - Cost-basis map (volume-by-price HVN levels — these are support/resistance zones)
  - Relative strength signal
  - The raw intrinsic-range price points (low / base / high) — JUST NUMBERS, no narrative

You DO NOT see the system's BUY/SELL/WAIT recommendation, the margin of safety
verdict, the company's fundamentals, or any other commentary. The point is to
produce an INDEPENDENT zone map that, when combined with the fundamental side
elsewhere, can either confirm or contradict the system's call.

Your output is a `PriceMap`:
  - 5-7 zones covering the range [intrinsic_low × 0.6, intrinsic_high × 1.2]
  - Each zone has price_low, price_high, label, and a ONE-SENTENCE rationale
  - Zones must be CONTIGUOUS and SORTED low→high
  - current_zone_index points to the zone containing current_price
  - key_support = the strongest HVN below current that is also a configured zone
                  boundary (lowest viable support)
  - key_resistance = the strongest HVN above current that is also a zone boundary

Labels available:
  aggressive_accumulation, accumulation, watch, hold, trim, distribution, high_risk

Use the cost-basis HVN locations as zone boundaries wherever possible — those
are observed market structure, not narrative.
"""


def _format_user_message(
    *,
    symbol: str,
    current_price: float,
    currency: str,
    intrinsic_low: float,
    intrinsic_base: float,
    intrinsic_high: float,
    structure: StructureAssessment,
    volume: VolumeAssessment,
    cost_basis: CostBasisMap,
    relative_strength: RelativeStrengthAssessment,
) -> str:
    hvn_lines = "\n".join(
        f"    {lv.price_low:.2f}–{lv.price_high:.2f} ({lv.volume_pct_of_window:.1f}% of volume, "
        f"{lv.position_vs_current} current, role={lv.role})"
        for lv in cost_basis.hvn_levels
    ) or "    (no high-volume nodes detected)"

    rs_summary = (
        f"vs sector 90d={relative_strength.vs_sector_etf_90d}, "
        f"vs sector 365d={relative_strength.vs_sector_etf_365d}, "
        f"vs index 90d={relative_strength.vs_index_90d}, "
        f"vs index 365d={relative_strength.vs_index_365d}, "
        f"signal={relative_strength.signal}"
    )

    return f"""TICKER: {symbol}  CURRENT: {current_price:.2f} {currency}

INTRINSIC RANGE (raw price points, no narrative):
  low = {intrinsic_low:.2f}
  base = {intrinsic_base:.2f}
  high = {intrinsic_high:.2f}

STRUCTURE:
  trend = {structure.trend}   stage = {structure.stage}   confidence = {structure.confidence:.2f}
  last_swing_low = {structure.last_swing_low:.2f}
  last_swing_high = {structure.last_swing_high:.2f}
  summary: {structure.structure_summary}

VOLUME:
  institutional_flow = {volume.institutional_flow}
  obv_trend = {volume.obv_trend}
  20d-vs-50d volume expansion: {volume.volume_expansion_pct:.1f}%
  up/down volume ratio: {volume.up_down_volume_ratio}
  signals: {", ".join(volume.signals) or "(none)"}

COST-BASIS (last {cost_basis.lookback_days}d):
  trapped supply above: {cost_basis.trapped_supply_pct:.1f}%
  accumulation below:   {cost_basis.accumulation_pct:.1f}%
  HVN levels:
{hvn_lines}

RELATIVE STRENGTH:
  {rs_summary}

Build the PriceMap per the system prompt.
"""


def _validate_price_map(pm: PriceMap, current_price: float) -> tuple[bool, Optional[str]]:
    """Cheap sanity checks: zones sorted, contiguous-ish, current_zone_index is valid."""
    if not pm.zones:
        return False, "zones is empty"
    for i in range(len(pm.zones) - 1):
        if pm.zones[i].price_high > pm.zones[i + 1].price_low + 0.01:
            return False, f"zones not sorted/non-overlapping at index {i}"
    if not (0 <= pm.current_zone_index < len(pm.zones)):
        return False, f"current_zone_index {pm.current_zone_index} out of range"
    zone = pm.zones[pm.current_zone_index]
    if not (zone.price_low - 0.5 <= current_price <= zone.price_high + 0.5):
        return False, f"current_price {current_price} not inside its zone [{zone.price_low}, {zone.price_high}]"
    return True, None


def _deterministic_fallback(
    *,
    current_price: float,
    intrinsic_low: float,
    intrinsic_base: float,
    intrinsic_high: float,
    cost_basis: CostBasisMap,
) -> PriceMap:
    """Build a Price Map from intrinsic range + HVN levels alone — no LLM."""
    supports = sorted(
        [lv for lv in cost_basis.hvn_levels if lv.position_vs_current == "below"],
        key=lambda x: x.price_high, reverse=True,
    )
    resistances = sorted(
        [lv for lv in cost_basis.hvn_levels if lv.position_vs_current == "above"],
        key=lambda x: x.price_low,
    )
    key_support = supports[0].price_high if supports else round(intrinsic_low * 0.85, 2)
    key_resistance = resistances[0].price_low if resistances else round(intrinsic_high * 1.10, 2)

    # 5 broad bands anchored on intrinsic + support
    zones = [
        PriceMapZone(price_low=round(intrinsic_low * 0.6, 2), price_high=round(intrinsic_low * 0.85, 2),
                     label="aggressive_accumulation", rationale="Deep below intrinsic low — capitulation range."),
        PriceMapZone(price_low=round(intrinsic_low * 0.85, 2), price_high=round(intrinsic_base * 0.95, 2),
                     label="accumulation", rationale="Below intrinsic base — value zone."),
        PriceMapZone(price_low=round(intrinsic_base * 0.95, 2), price_high=round(intrinsic_base * 1.05, 2),
                     label="hold", rationale="Around fair value — neither cheap nor expensive."),
        PriceMapZone(price_low=round(intrinsic_base * 1.05, 2), price_high=round(intrinsic_high * 1.05, 2),
                     label="trim", rationale="Above intrinsic base toward intrinsic high — reduce."),
        PriceMapZone(price_low=round(intrinsic_high * 1.05, 2), price_high=round(intrinsic_high * 1.20, 2),
                     label="distribution", rationale="Beyond intrinsic high — distribute."),
    ]
    # Find current zone
    current_idx = 0
    for i, z in enumerate(zones):
        if z.price_low <= current_price <= z.price_high:
            current_idx = i
            break
    else:
        # If current_price falls outside, pick the closest endpoint
        if current_price < zones[0].price_low:
            current_idx = 0
        else:
            current_idx = len(zones) - 1
            # Extend the top zone upward to cover the current price
            zones[-1] = PriceMapZone(
                price_low=zones[-1].price_low,
                price_high=max(zones[-1].price_high, current_price + 1.0),
                label=zones[-1].label, rationale=zones[-1].rationale,
            )

    return PriceMap(
        zones=zones,
        current_zone_index=current_idx,
        key_support=round(key_support, 2),
        key_resistance=round(key_resistance, 2),
        summary="Deterministic fallback price map built from intrinsic range + cost-basis HVNs.",
    )


def build_price_map(
    *,
    symbol: str,
    current_price: float,
    currency: str,
    intrinsic_low: float,
    intrinsic_base: float,
    intrinsic_high: float,
    structure: StructureAssessment,
    volume: VolumeAssessment,
    cost_basis: CostBasisMap,
    relative_strength: RelativeStrengthAssessment,
    enable_llm: bool = True,
) -> PriceMap:
    if not enable_llm:
        return _deterministic_fallback(
            current_price=current_price,
            intrinsic_low=intrinsic_low, intrinsic_base=intrinsic_base, intrinsic_high=intrinsic_high,
            cost_basis=cost_basis,
        )

    user_msg = _format_user_message(
        symbol=symbol, current_price=current_price, currency=currency,
        intrinsic_low=intrinsic_low, intrinsic_base=intrinsic_base, intrinsic_high=intrinsic_high,
        structure=structure, volume=volume, cost_basis=cost_basis,
        relative_strength=relative_strength,
    )
    try:
        pm = chat_json(system=SYSTEM_PROMPT, user=user_msg, schema=PriceMap, temperature=0.2)
        ok, msg = _validate_price_map(pm, current_price)
        if not ok:
            log.warning("Price Map LLM output failed validation: %s — using deterministic fallback.", msg)
            return _deterministic_fallback(
                current_price=current_price,
                intrinsic_low=intrinsic_low, intrinsic_base=intrinsic_base, intrinsic_high=intrinsic_high,
                cost_basis=cost_basis,
            )
        return pm
    except Exception as e:
        log.warning("Price Map LLM call failed (%s) — using deterministic fallback.", e)
        return _deterministic_fallback(
            current_price=current_price,
            intrinsic_low=intrinsic_low, intrinsic_base=intrinsic_base, intrinsic_high=intrinsic_high,
            cost_basis=cost_basis,
        )
