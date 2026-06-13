"""Hedging agent — picks futures / ETF-short hedge candidates by correlation.

Runs only when Tactical Exit recommends a hedge (RED_DEFENSIVE level or higher
with hedge_remainder=true). Inputs: position to hedge, sector/market, position
value. Outputs ranked HedgeCandidates with the recommended one flagged.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.data.futures import HedgeInstrument, candidate_pool, compute_correlations
from src.llm.client import chat_json
from src.models.schemas import HedgeCandidate, HedgePlan

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a derivatives strategist designing a hedge for a long equity position.

You receive:
  - The position to hedge (symbol, market, sector, USD-equivalent value)
  - A candidate pool of futures and short-ETF instruments
  - The 90-day correlation of each candidate to the position's daily returns

Your job: rank the candidates and recommend ONE.

Decision criteria (in order):
  1. CORRELATION EFFECTIVENESS — higher |correlation| = better hedge. Stop-considering candidates with |corr| < 0.3.
  2. INSTRUMENT FIT to position size:
       - Futures are leveraged. For positions below ~$100k, a single contract often overshoots.
       - Use micro-futures (MES, MNQ) or short-ETFs for smaller books.
  3. LIQUIDITY — ES, NQ, HSI are top tier. Sector ETFs vary.
  4. BASIS RISK — sector ETF on US Tech vs the same stock has less basis risk than broad S&P futures.

Output (HedgePlan schema):
  - candidates: list of HedgeCandidate, ordered best-to-worst. Include ALL eligible
    candidates (|corr| ≥ 0.3) plus the top 1-2 ineligible ones with brief reason.
  - For each: instrument, instrument_kind, correlation_90d, rationale (1 sentence),
    suggested_notional_usd (approximate dollar exposure to neutralize position).
    - For FUTURES: suggested_notional_usd ≈ position_value × |corr|. Round to whole contracts using contract_notional.
    - For ETF SHORTS: same — short ETF shares worth that notional.
  - recommended_index: index in `candidates` of the recommended one. Choose by criteria 1-4 above.
  - rationale: 2-3 sentences on WHY you chose it (correlation, fit, liquidity).
  - notes: caveats — margin requirements, expiry roll dates, basis risk reminders.

Per the investment framework (Ch.3): leverage amplifies death rates. Be conservative
about hedge sizing. Prefer 60-80% notional coverage to a 100% naked hedge that introduces
its own tail risk.
"""


def _format_user(
    symbol: str,
    market: str,
    sector: Optional[str],
    position_value_usd: float,
    candidates: list[HedgeInstrument],
    correlations: dict[str, Optional[float]],
    prefer_etf_short_below_usd: Optional[float] = None,
) -> str:
    cand_lines = "\n".join(
        f"  {i+1}. {c.ticker} ({c.kind}): {c.description}\n"
        f"        90d correlation to {symbol}: {correlations.get(c.ticker)}\n"
        f"        approx notional per contract: ${c.notional_per_contract_usd:,.0f}" if c.notional_per_contract_usd
        else f"  {i+1}. {c.ticker} ({c.kind}): {c.description}\n"
        f"        90d correlation to {symbol}: {correlations.get(c.ticker)}\n"
        f"        ETF — size by USD notional"
        for i, c in enumerate(candidates)
    )

    pref_clause = ""
    if prefer_etf_short_below_usd is not None and position_value_usd < prefer_etf_short_below_usd:
        pref_clause = (
            f"\nSIZING PREFERENCE: position (${position_value_usd:,.0f}) is below "
            f"the ${prefer_etf_short_below_usd:,.0f} threshold for ETF-short preference. "
            f"Prefer an ETF-short candidate over a futures candidate UNLESS a futures "
            f"candidate has materially better correlation (|corr| advantage ≥ 0.15)."
        )

    return f"""POSITION TO HEDGE:
  Symbol:       {symbol}
  Market:       {market}
  Sector:       {sector or 'unknown'}
  Value (USD):  ${position_value_usd:,.0f}
{pref_clause}

CANDIDATE POOL ({len(candidates)} instruments):
{cand_lines}

Rank these and recommend ONE, per the system prompt. Sized for the position's notional.
"""


def design_hedge(
    symbol: str,
    market: str,
    sector: Optional[str],
    position_value_usd: float,
    prefer_etf_short_below_usd: Optional[float] = None,
) -> HedgePlan:
    """Pick candidate pool, compute correlations, ask LLM to rank."""
    candidates = candidate_pool(market, sector)
    if not candidates:
        return HedgePlan(
            symbol_being_hedged=symbol,
            position_value_usd=position_value_usd,
            candidates=[],
            recommended_index=0,
            rationale="No candidate hedges available for this market/sector combination.",
            notes=[],
        )

    log.info("Computing 90d correlations for %d hedge candidates against %s…", len(candidates), symbol)
    corrs = compute_correlations(symbol, candidates, period_days=90)

    user_msg = _format_user(
        symbol, market, sector, position_value_usd, candidates, corrs,
        prefer_etf_short_below_usd=prefer_etf_short_below_usd,
    )
    plan = chat_json(
        system=SYSTEM_PROMPT,
        user=user_msg,
        schema=HedgePlan,
        temperature=0.2,
    )

    # Force these to match inputs
    plan.symbol_being_hedged = symbol
    plan.position_value_usd = position_value_usd

    # Backfill correlations on the candidates the LLM returned (it may forget to)
    for cand in plan.candidates:
        if cand.correlation_90d is None and cand.instrument in corrs:
            cand.correlation_90d = corrs[cand.instrument]

    return plan
