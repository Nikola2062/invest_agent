"""Forward Scenario agent — explicit probability-weighted price paths.

Consumes:
  - ForwardCatalysts (from IR agent)
  - Current price + intrinsic valuation range
  - Macro snapshot (regime context)

Produces:
  - ForwardScenarios: 3-4 named price paths with low/base/high target prices,
    estimated drawdown during the path, and key drivers. Probability-weighted
    expected return.

This is DIFFERENT from the Risk Analyzer's `scenarios` field. Risk Analyzer outputs
drawdown probability distribution (used by Tactical Exit). Forward Scenario outputs
explicit PRICE TARGETS at the horizon (used for decision-making + dashboard display).
"""
from __future__ import annotations

from typing import Optional

from src.agents.base import fmt_optional
from src.data.macro import MacroSnapshot
from src.llm.client import chat_json
from src.models.schemas import (
    ForwardCatalysts,
    ForwardScenarios,
    PriceScenario,
    ValuationResult,
)


SYSTEM_PROMPT = """You are a research strategist producing PROBABILITY-WEIGHTED PRICE PATHS at a fixed horizon.

You receive:
  - Current price + currency
  - Intrinsic valuation range (low/base/high)
  - Forward catalysts with dates / direction / magnitude / confidence
  - Macro context (yield curve, VIX, S&P/HSI drawdowns, key signals)

Your job: produce 3-4 NAMED SCENARIOS at the horizon. Probabilities MUST sum to 1.0.

Naming convention (use these unless you have a strong reason for different names):
  - "base"        ≈ 50-65% probability — most likely outcome
  - "bull"        ≈ 15-25% — upside scenario
  - "bear"        ≈ 10-20% — downside scenario
  - "black_swan"  ≈  2-8%  — tail-risk scenario (optional but recommended if there's identifiable tail risk)

Each scenario MUST include:
  - target_price_low / target_price_base / target_price_high at the horizon
       These are the price RANGE for that scenario (not the same as intrinsic range).
       E.g. base case: $90 - $100 - $110.  Bear case: $60 - $70 - $80.
  - return_pct_base = (target_price_base / current_price - 1) * 100
  - drawdown_pct_estimated: maximum drawdown DURING the path (negative number).
       Bull case may still draw down 10%; bear case will draw down more.
  - key_drivers: 2-4 specific things (catalysts, macro events, regime shifts) that drive this scenario
  - rationale: 1-2 sentences

After listing scenarios:
  - probability_weighted_target = Σ (probability × target_price_base)
  - expected_return_pct = (probability_weighted_target / current_price - 1) × 100
  - summary: 1-2 sentences. Honest. If the expected value is barely above 0%, say so.

Discipline:
  - Anchor on intrinsic value. If intrinsic_base is $100, scenarios should converge there in 80%+ of cases unless a catalyst materially shifts intrinsic.
  - Don't construct scenarios that all skew bullish OR all skew bearish to fit a narrative.
  - Black-swan scenarios are rare BY DEFINITION. Don't include one if there's no identifiable tail.
  - Be calibrated to history: equities typically return 5-10% annualized. A "base case" of +40% in 90 days is suspicious unless backed by a specific catalyst.
"""


def _format_user(
    symbol: str,
    current_price: float,
    currency: str,
    valuation: ValuationResult,
    forward_catalysts: Optional[ForwardCatalysts],
    macro: MacroSnapshot,
    horizon_days: int,
) -> str:
    cats_block = "  (no forward catalysts available)"
    if forward_catalysts and forward_catalysts.key_catalysts:
        cats_block = "\n".join(
            f"  - {c.event} [{c.direction}/{c.confidence}/~{c.expected_magnitude_pct or '?'}%] "
            f"{c.expected_date.isoformat() if c.expected_date else 'no date'}: {c.rationale}"
            for c in forward_catalysts.key_catalysts[:8]
        )

    macro_block = "\n".join(f"  - {s}" for s in macro.signals) or "  (no stress signals firing)"

    return f"""SYMBOL: {symbol}
CURRENT PRICE: {current_price:.2f} {currency}
HORIZON: {horizon_days} days

INTRINSIC VALUATION:
  Range: {valuation.intrinsic_low:.2f} – {valuation.intrinsic_base:.2f} – {valuation.intrinsic_high:.2f}
  MoS:   {valuation.margin_of_safety_pct:+.1f}%
  Confidence: {valuation.confidence}

FORWARD CATALYSTS (from IR agent):
{cats_block}

ACTIVE MACRO SIGNALS:
{macro_block}

MACRO LEVELS:
  10Y-3M spread:       {fmt_optional(macro.yield_curve_10y3m_pct, '{:.2f}%')}
  VIX:                 {fmt_optional(macro.vix_level)}
  S&P 500 vs 52w high: {fmt_optional(macro.sp500_drawdown_pct, '{:.1f}%')}
  Hang Seng vs 52w high: {fmt_optional(macro.hsi_drawdown_pct, '{:.1f}%')}

Produce 3-4 probability-weighted price scenarios at the horizon per the system prompt.
Compute probability_weighted_target and expected_return_pct from the scenarios you output.
"""


def generate_scenarios(
    symbol: str,
    current_price: float,
    currency: str,
    valuation: ValuationResult,
    macro: MacroSnapshot,
    horizon_days: int = 90,
    forward_catalysts: Optional[ForwardCatalysts] = None,
) -> ForwardScenarios:
    user_msg = _format_user(
        symbol, current_price, currency, valuation, forward_catalysts, macro, horizon_days
    )
    # temperature=0: scenario targets and expected return drive downstream metrics
    # that should be stable across same-day reruns.
    result = chat_json(
        system=SYSTEM_PROMPT,
        user=user_msg,
        schema=ForwardScenarios,
        temperature=0.0,
    )

    # Force the meta fields to match inputs exactly
    result.symbol = symbol
    result.current_price = current_price
    result.currency = currency
    result.horizon_days = horizon_days

    # Recompute weighted target deterministically (don't trust the LLM's arithmetic)
    if result.scenarios:
        # Normalize probabilities if they don't sum to 1
        total_p = sum(s.probability for s in result.scenarios)
        if total_p > 0 and abs(total_p - 1.0) > 0.01:
            for s in result.scenarios:
                s.probability /= total_p
        weighted = sum(s.probability * s.target_price_base for s in result.scenarios)
        result.probability_weighted_target = weighted
        result.expected_return_pct = (weighted / current_price - 1.0) * 100.0

    return result
