"""Order Generator agent — translates decisions into concrete order specs.

Two paths:
  1. If held: tactical exit recommendations → immediate SELL orders + conditional rebuy BUY orders.
  2. If not held: valuation-based entry decision → BUY_NOW or WAIT_FOR_PRICE or PASS, with concrete entry orders.

Pure deterministic. No LLM.
"""
from __future__ import annotations

from typing import Optional

from src.config.loader import load_valuation
from src.models.schemas import (
    HeldDecision,
    Holding,
    NotHeldDecision,
    OrderSpec,
    TacticalAction,
    ValuationResult,
)


# --- IF HELD ---

def generate_held_orders(
    holding: Holding,
    current_price: float,
    tactical: TacticalAction,
) -> HeldDecision:
    immediate: list[OrderSpec] = []
    rebuy: list[OrderSpec] = []

    if tactical.action in ("trim", "defensive_reduction", "full_exit") and tactical.trim_pct_of_position:
        trim_shares = round(holding.shares * tactical.trim_pct_of_position / 100.0)
        if trim_shares > 0:
            # SELL at limit slightly above current to encourage fill but capture spread.
            sell_limit = round(current_price * 0.998, 4)
            immediate.append(
                OrderSpec(
                    side="SELL",
                    quantity=trim_shares,
                    symbol=holding.symbol,
                    order_type="LIMIT",
                    limit_price=sell_limit,
                    time_in_force="DAY",
                    rationale=(
                        f"Defensive reduction per {tactical.label}: trim {tactical.trim_pct_of_position:.0f}% "
                        f"of position ({holding.shares:.0f} → {holding.shares - trim_shares:.0f} shares)."
                    ),
                )
            )

            # Rebuy plan: split trimmed shares across the band (50/50 between top and bottom of band).
            if tactical.rebuy_band_high and tactical.rebuy_band_low:
                rebuy_high_price = round(tactical.rebuy_band_high, 4)
                rebuy_low_price = round(tactical.rebuy_band_low, 4)
                first_tranche = trim_shares // 2
                second_tranche = trim_shares - first_tranche
                if first_tranche > 0:
                    rebuy.append(
                        OrderSpec(
                            side="BUY",
                            quantity=first_tranche,
                            symbol=holding.symbol,
                            order_type="LIMIT",
                            limit_price=rebuy_high_price,
                            time_in_force="GTC",
                            conditional=True,
                            rationale=(
                                f"Tranche 1 of pre-committed rebuy: fills if price drops to "
                                f"${rebuy_high_price:.2f} (top of band)."
                            ),
                        )
                    )
                if second_tranche > 0:
                    rebuy.append(
                        OrderSpec(
                            side="BUY",
                            quantity=second_tranche,
                            symbol=holding.symbol,
                            order_type="LIMIT",
                            limit_price=rebuy_low_price,
                            time_in_force="GTC",
                            conditional=True,
                            rationale=(
                                f"Tranche 2 of pre-committed rebuy: fills if price drops to "
                                f"${rebuy_low_price:.2f} (bottom of band)."
                            ),
                        )
                    )

    return HeldDecision(tactical=tactical, immediate_orders=immediate, rebuy_orders=rebuy)


# --- IF NOT HELD ---

def generate_entry_orders(
    symbol: str,
    market: str,
    current_price: float,
    valuation: ValuationResult,
    available_cash: Optional[float] = None,
    default_position_size_pct: Optional[float] = None,
) -> NotHeldDecision:
    """Build a not-held BUY decision driven by margin of safety thresholds.

    Thresholds (MoS bar, wait-band discount, default position size) live in
    config/valuation.yaml::entry_decision. Pass-through args override the config
    for tests and one-off callers.
    """
    entry_cfg = load_valuation().entry_decision
    mos_bar = entry_cfg.margin_of_safety_required_pct
    wait_discount = entry_cfg.wait_band_discount_pct
    size_pct = default_position_size_pct if default_position_size_pct is not None else entry_cfg.default_position_size_pct
    mos = valuation.margin_of_safety_pct

    # If we don't know cash, suggest 100 shares as a placeholder qty — user adjusts.
    if available_cash is not None and available_cash > 0:
        target_size_value = available_cash * size_pct / 100.0
        target_qty = max(1, int(target_size_value / current_price))
    else:
        target_qty = 100

    if mos >= mos_bar and valuation.confidence != "low":
        return NotHeldDecision(
            recommendation="BUY_NOW",
            entry_orders=[
                OrderSpec(
                    side="BUY",
                    quantity=target_qty,
                    symbol=symbol,
                    order_type="LIMIT",
                    limit_price=round(current_price * 1.005, 4),
                    time_in_force="DAY",
                    rationale=(
                        f"Margin of safety {mos:.1f}% (>= {mos_bar:.0f}% bar). "
                        f"Confidence: {valuation.confidence}. Buy at small premium to current to ensure fill."
                    ),
                )
            ],
            rationale=(
                f"Current price ${current_price:.2f} is {mos:.1f}% below intrinsic base "
                f"${valuation.intrinsic_base:.2f}. Meets MoS bar with {valuation.confidence} confidence."
            ),
        )

    if mos > -10 and valuation.confidence != "low":
        # Within 10% of fair value — wait for a better price.
        wait_price = round(valuation.intrinsic_low * (1 - wait_discount / 100.0), 4)
        # Only if wait_price is below current
        if wait_price < current_price:
            return NotHeldDecision(
                recommendation="WAIT_FOR_PRICE",
                entry_orders=[
                    OrderSpec(
                        side="BUY",
                        quantity=target_qty,
                        symbol=symbol,
                        order_type="LIMIT",
                        limit_price=wait_price,
                        time_in_force="GTC",
                        conditional=True,
                        rationale=(
                            f"Place resting limit at ${wait_price:.2f} (intrinsic_low ${valuation.intrinsic_low:.2f} "
                            f"minus {wait_discount:.0f}% extra cushion)."
                        ),
                    )
                ],
                rationale=(
                    f"MoS {mos:.1f}% — not enough for immediate buy. Waiting for ${wait_price:.2f} "
                    f"would give margin of safety closer to {(valuation.intrinsic_base / wait_price - 1) * 100:.1f}%."
                ),
            )

    return NotHeldDecision(
        recommendation="PASS",
        entry_orders=[],
        rationale=(
            f"MoS {mos:.1f}% does not meet requirement, or confidence is too low "
            f"({valuation.confidence}). Pass for now."
        ),
    )
