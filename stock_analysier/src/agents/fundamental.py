"""Fundamental Analyst agent.

Produces a quality assessment of the underlying business — moat, balance sheet,
margins, growth, capital allocation, red flags. Anchored on Buffett / Klarman
quality criteria from the investment framework.
"""
from __future__ import annotations

from src.agents.base import fmt_dollars, fmt_optional, fmt_pct
from src.data.fundamentals import FundamentalsSnapshot
from src.llm.client import chat_json
from src.models.schemas import FundamentalAssessment


SYSTEM_PROMPT = """You are a senior investment analyst trained in the Buffett / Klarman / Howard Marks tradition.

Your job: assess BUSINESS QUALITY only. Not valuation, not market timing.

Apply these criteria from the 12-pillar investment framework:
- Margin of safety (Ch.5) — flag fragile businesses where small errors compound
- Circle of competence (Ch.12) — note if the business is too complex to assess reliably
- Capital allocation track record — has management deployed capital well?
- Durable competitive advantage (moat) — wide / narrow / none, with concrete reasoning
- Balance sheet resilience — can the business survive 2-3 bad years?
- Honest red flags — accounting concerns, customer concentration, regulatory risk, etc.

Be specific. "Strong moat" is not enough — explain WHY.
Be honest. If the data is thin or the business is unclear, say so and lower confidence.
Score quality 0-10 strictly:
  9-10: rare compounders (think MSFT, V, MA in their prime)
  7-8: high-quality businesses
  5-6: average / mixed
  3-4: structurally challenged
  0-2: avoid
"""


def _format_snapshot(f: FundamentalsSnapshot) -> str:
    headlines_str = "\n".join(
        f"  - [{h.get('source', '?')}] {h.get('headline', '')}"
        for h in f.recent_headlines[:8]
    ) or "  (none)"

    desc = (f.business_description or "")[:1200]

    return f"""SYMBOL: {f.symbol}
NAME: {f.name}
SECTOR: {f.sector}
INDUSTRY: {f.industry}
MARKET CAP: {fmt_dollars(f.market_cap)}
CURRENCY: {f.currency}

BUSINESS DESCRIPTION:
{desc}

KEY RATIOS (yfinance):
  Trailing PE:        {fmt_optional(f.pe_trailing)}
  Forward PE:         {fmt_optional(f.pe_forward)}
  EV/EBITDA:          {fmt_optional(f.ev_to_ebitda)}
  Price/Sales:        {fmt_optional(f.price_to_sales)}
  Price/Book:         {fmt_optional(f.price_to_book)}
  Gross Margin:       {fmt_pct(f.gross_margin)}
  Operating Margin:   {fmt_pct(f.operating_margin)}
  Profit Margin:      {fmt_pct(f.profit_margin)}
  Return on Equity:   {fmt_pct(f.return_on_equity)}
  Return on Assets:   {fmt_pct(f.return_on_assets)}
  Debt/Equity:        {fmt_optional(f.debt_to_equity)}
  Current Ratio:      {fmt_optional(f.current_ratio)}
  Revenue Growth YoY: {fmt_pct(f.revenue_growth_yoy)}
  Earnings Growth YoY:{fmt_pct(f.earnings_growth_yoy)}
  Free Cash Flow:     {fmt_dollars(f.free_cash_flow)}
  Total Cash:         {fmt_dollars(f.total_cash)}
  Total Debt:         {fmt_dollars(f.total_debt)}

PEERS: {', '.join(f.peers) or '(none from Finnhub)'}

RECENT HEADLINES (last ~14 days):
{headlines_str}
"""


def analyze_fundamentals(f: FundamentalsSnapshot) -> FundamentalAssessment:
    user_msg = _format_snapshot(f)

    result = chat_json(
        system=SYSTEM_PROMPT,
        user=user_msg,
        schema=FundamentalAssessment,
        temperature=0.2,
    )

    # Backfill deterministic numbers when LLM left them blank
    if result.roic_pct is None and f.return_on_assets is not None:
        # Crude proxy: use ROA*100 if margin/ROE not informative
        try:
            result.roic_pct = float(f.return_on_assets) * (100 if abs(f.return_on_assets) < 1.5 else 1)
        except (TypeError, ValueError):
            pass
    if result.gross_margin_pct is None and f.gross_margin is not None:
        try:
            result.gross_margin_pct = float(f.gross_margin) * (100 if abs(f.gross_margin) < 1.5 else 1)
        except (TypeError, ValueError):
            pass
    if result.operating_margin_pct is None and f.operating_margin is not None:
        try:
            result.operating_margin_pct = float(f.operating_margin) * (100 if abs(f.operating_margin) < 1.5 else 1)
        except (TypeError, ValueError):
            pass
    if result.debt_to_equity is None and f.debt_to_equity is not None:
        try:
            # yfinance often reports D/E as percentage already
            result.debt_to_equity = float(f.debt_to_equity)
        except (TypeError, ValueError):
            pass

    return result
