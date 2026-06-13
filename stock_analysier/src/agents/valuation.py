"""Valuation agent.

1. Deterministic: simple 2-stage DCF + multiples-based fair value.
2. LLM: reviews the numbers, applies judgment, produces intrinsic value range
   (low/base/high) and confidence rating.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from src.agents.base import fmt_dollars, fmt_optional
from src.config.loader import load_valuation
from src.data.fundamentals import FundamentalsSnapshot
from src.data.prices import PriceSnapshot
from src.llm.client import chat_json
from src.models.schemas import (
    FundamentalAssessment,
    ValuationResult,
)


SYSTEM_PROMPT = """You are a senior equity valuation analyst.

Your job: produce an INTRINSIC VALUE RANGE per share — low / base / high — and a confidence rating.

Inputs you receive:
  - Current price, currency
  - A deterministic 2-stage DCF and a multiples-based fair value (best-effort, may be missing)
  - Fundamental ratios and the quality assessment from the Fundamental Analyst
  - Peer set

Rules:
  - Anchor on the deterministic numbers BUT apply judgment. If the DCF assumptions are clearly off
    (e.g., unrealistic growth, negative FCF, early-stage company), say so and override.
  - "Margin of safety" (vs intrinsic_base) is computed downstream — you only output the range.
  - For unprofitable / pre-FCF businesses, lean on revenue multiples vs peers and explain.
  - Be honest about uncertainty. Confidence:
      high   = mature business with stable FCF and clear comps; narrow range
      medium = decent visibility but some open questions; moderate range
      low    = early stage / no profits / heavy dependence on one assumption; wide range
  - methodology_notes: 2-4 sentences. State the dominant approach (DCF / multiples / asset-based / sum-of-parts) and the key assumption that drives the range.
"""


@dataclass
class DCFInputs:
    fcf_per_share_ttm: Optional[float]
    growth_rate_y1_5: float  # decimal
    terminal_growth: float   # decimal
    discount_rate: float     # decimal
    shares_outstanding: Optional[float]


def _two_stage_dcf(inputs: DCFInputs) -> Optional[float]:
    """Returns intrinsic per-share value, or None if inputs are insufficient."""
    if inputs.fcf_per_share_ttm is None or inputs.fcf_per_share_ttm <= 0:
        return None
    if inputs.discount_rate <= inputs.terminal_growth:
        return None  # math breaks
    fcf = inputs.fcf_per_share_ttm
    g1 = inputs.growth_rate_y1_5
    g_term = inputs.terminal_growth
    r = inputs.discount_rate

    pv = 0.0
    for year in range(1, 6):
        fcf_y = fcf * (1 + g1) ** year
        pv += fcf_y / (1 + r) ** year

    # Terminal value at end of year 5
    fcf_y6 = fcf * (1 + g1) ** 5 * (1 + g_term)
    terminal_value = fcf_y6 / (r - g_term)
    pv += terminal_value / (1 + r) ** 5
    return pv


def _multiples_value(f: FundamentalsSnapshot, current_price: float) -> Optional[float]:
    """Crude average across trailing PE, forward PE, EV/EBITDA implied prices.

    For each, if a peer set existed we'd compare; without that, we use the
    company's own multiples to back into an implied EPS-or-EBITDA-derived value
    that the LLM can sanity check.
    """
    candidates = []

    # Forward PE × EPS implied (uses trailingEPS proxy)
    if f.pe_forward and f.pe_forward > 0 and f.raw_info.get("forwardEps"):
        try:
            candidates.append(float(f.pe_forward) * float(f.raw_info["forwardEps"]))
        except (TypeError, ValueError):
            pass
    if f.pe_trailing and f.pe_trailing > 0 and f.raw_info.get("trailingEps"):
        try:
            candidates.append(float(f.pe_trailing) * float(f.raw_info["trailingEps"]))
        except (TypeError, ValueError):
            pass

    if not candidates:
        return None
    return float(sum(candidates) / len(candidates))


def _estimate_dcf(f: FundamentalsSnapshot, market: Optional[str] = None) -> tuple[Optional[float], DCFInputs]:
    """Build DCF inputs using config/valuation.yaml defaults, optionally
    overridden per market (US gets a lower discount rate, HK a higher one)."""
    cfg = load_valuation().dcf.resolved(market)

    shares = f.raw_info.get("sharesOutstanding") or f.raw_info.get("impliedSharesOutstanding")

    fcf_per_share = None
    if f.free_cash_flow and shares:
        try:
            fcf_per_share = float(f.free_cash_flow) / float(shares)
        except (TypeError, ValueError):
            fcf_per_share = None

    # Growth: use earnings_growth or revenue_growth, cap at sensible bounds.
    g = f.earnings_growth_yoy if f.earnings_growth_yoy is not None else f.revenue_growth_yoy
    if g is None:
        g_y1_5 = cfg.fallback_growth_y1_5
    else:
        try:
            g_dec = float(g) if abs(float(g)) < 1.5 else float(g) / 100.0
            g_y1_5 = max(cfg.growth_floor_y1_5, min(cfg.growth_cap_y1_5, g_dec))
        except (TypeError, ValueError):
            g_y1_5 = cfg.fallback_growth_y1_5

    inputs = DCFInputs(
        fcf_per_share_ttm=fcf_per_share,
        growth_rate_y1_5=g_y1_5,
        terminal_growth=cfg.terminal_growth,
        discount_rate=cfg.discount_rate,
        shares_outstanding=float(shares) if shares else None,
    )
    return _two_stage_dcf(inputs), inputs


def _format_user(
    f: FundamentalsSnapshot,
    p: PriceSnapshot,
    fa: FundamentalAssessment,
    dcf_value: Optional[float],
    dcf_inputs: DCFInputs,
    multiples_value: Optional[float],
) -> str:
    return f"""SYMBOL: {f.symbol}
CURRENT PRICE: {p.current_price:.2f} {p.currency}
MARKET CAP: {fmt_dollars(f.market_cap)}

DETERMINISTIC DCF (2-stage, 5yr explicit + perpetuity):
  Inputs:
    FCF per share (TTM):  {fmt_optional(dcf_inputs.fcf_per_share_ttm, '${:.2f}')}
    Growth y1-5:          {dcf_inputs.growth_rate_y1_5 * 100:.1f}%
    Terminal growth:      {dcf_inputs.terminal_growth * 100:.1f}%
    Discount rate:        {dcf_inputs.discount_rate * 100:.1f}%
    Shares outstanding:   {fmt_dollars(dcf_inputs.shares_outstanding)}
  Result (per share):     {fmt_optional(dcf_value, '${:.2f}')}

MULTIPLES-BASED FAIR VALUE (own-PE blend):
  Result (per share):     {fmt_optional(multiples_value, '${:.2f}')}

KEY RATIOS:
  Trailing PE: {fmt_optional(f.pe_trailing)}    Forward PE: {fmt_optional(f.pe_forward)}
  EV/EBITDA:   {fmt_optional(f.ev_to_ebitda)}   P/S:        {fmt_optional(f.price_to_sales)}
  Op Margin:   {fmt_optional(f.operating_margin)}  ROE:     {fmt_optional(f.return_on_equity)}
  Debt/Equity: {fmt_optional(f.debt_to_equity)}

QUALITY ASSESSMENT (from Fundamental Analyst):
  Quality score: {fa.quality_score:.1f}/10
  Moat:          {fa.moat_strength}
  Balance sheet: {fa.balance_sheet_health}
  Thesis:        {fa.thesis_one_liner}
  Red flags:     {'; '.join(fa.red_flags) or '(none)'}

PEERS: {', '.join(f.peers) or '(none)'}

PRICE CONTEXT:
  Realized vol (annualized): {p.realized_vol_annualized_pct:.1f}%
  1m / 3m / 1y return:        {p.cumulative_return_1m_pct:.1f}% / {p.cumulative_return_3m_pct:.1f}% / {p.cumulative_return_1y_pct:.1f}%

Output an intrinsic value range. Use the deterministic numbers as anchors but apply judgment.
The current_price and currency in your output should match the inputs exactly.
"""


def analyze_valuation(
    f: FundamentalsSnapshot,
    p: PriceSnapshot,
    fundamental: FundamentalAssessment,
    market: Optional[str] = None,
) -> ValuationResult:
    dcf_value, dcf_inputs = _estimate_dcf(f, market=market)
    multiples_value = _multiples_value(f, p.current_price)

    user_msg = _format_user(f, p, fundamental, dcf_value, dcf_inputs, multiples_value)
    # temperature=0: this output feeds the deterministic tactical ladder and the
    # dashboard's single-point MoS. Same-day reruns must produce the same intrinsic.
    result = chat_json(
        system=SYSTEM_PROMPT,
        user=user_msg,
        schema=ValuationResult,
        temperature=0.0,
    )

    # Backfill numerics the LLM might not have echoed
    if result.dcf_value is None:
        result.dcf_value = dcf_value
    if result.multiples_value is None:
        result.multiples_value = multiples_value
    if result.pe_ratio is None:
        result.pe_ratio = f.pe_trailing
    if result.ev_to_ebitda is None:
        result.ev_to_ebitda = f.ev_to_ebitda
    result.current_price = p.current_price
    result.currency = p.currency

    # Compute margin of safety vs intrinsic_base, if not already meaningful
    if result.intrinsic_base > 0:
        result.margin_of_safety_pct = (result.intrinsic_base / p.current_price - 1.0) * 100.0

    return result
