"""Information Retrieval agent.

Consumes:
  - CatalystSnapshot (deterministic: earnings calendar, economic calendar, recent news)
  - Stock context: symbol, sector, business summary

Produces:
  - ForwardCatalysts: 5-10 distilled forward-looking catalysts with direction, magnitude, confidence
  - macro_overlay: relevant macro events
  - sentiment_summary + score from recent news

This feeds into the Risk Analyzer.
"""
from __future__ import annotations

from datetime import date

from src.data.catalysts import CatalystSnapshot
from src.llm.client import chat_json
from src.models.schemas import ForwardCatalysts


SYSTEM_PROMPT = """You are a research analyst tracking forward-looking events that could move a single stock.

Your job: distill RAW catalyst data into 5-10 KEY catalysts with a clear direction, magnitude, and confidence.

Inputs you'll receive:
  - Symbol + sector + brief business description
  - Upcoming earnings dates with consensus estimates
  - Upcoming economic events (Fed meetings, CPI, NFP, central bank decisions, etc.)
  - Recent news headlines (last ~14 days, mixed signal-to-noise)

Output:
  - key_catalysts: 5-10 items, each with:
      event:        concise (≤ 12 words)
      expected_date: ISO date if known
      direction:    "positive" / "negative" / "uncertain"
      expected_magnitude_pct: rough % impact on stock on resolution (e.g. 3.0 for ±3%). Conservative.
      confidence:   "high" / "medium" / "low"
      rationale:    1 sentence
  - macro_overlay: 3-5 macro events likely to affect THIS stock specifically (e.g. Fed rate decision for rate-sensitive names; PBoC stimulus for HK names).
  - sentiment_summary: 1-2 sentences. Honest read of the news.
  - sentiment_score: -1 (very bearish) to +1 (very bullish). 0 = neutral.

Discipline:
  - Filter aggressively. Most news is noise. Ignore SEO/job listings, generic "Show HN" posts, listicles.
  - Be honest about uncertainty — most "catalysts" are low-confidence by their nature.
  - Don't fabricate events that aren't in the data. If there are no catalysts in horizon, return an empty list.
  - Distinguish between EXISTING news (already priced in) and UPCOMING events (not yet priced).
"""


def _format_input(c: CatalystSnapshot, sector: str | None, business_summary: str | None) -> str:
    earnings_lines = "\n".join(
        f"  - {e.event_date.isoformat()} {e.hour or ''} EPS est={e.eps_estimate} Rev est={e.revenue_estimate}"
        for e in c.upcoming_earnings
    ) or "  (none in horizon)"

    econ_lines = "\n".join(
        f"  - {e.event_date.isoformat()} [{e.impact}] {e.country} — {e.event} (est={e.estimate}, prev={e.prev})"
        for e in c.upcoming_economic_events[:15]
    ) or "  (none high/medium impact in horizon)"

    news_lines = "\n".join(
        f"  - {n.timestamp_utc.date()} [{n.source}] {n.headline}"
        + (f" — {n.summary[:140]}" if n.summary else "")
        for n in c.recent_news[:20]
    ) or "  (none)"

    return f"""SYMBOL: {c.symbol}
MARKET: {c.market}
HORIZON: {c.horizon_days} days
SECTOR: {sector or 'unknown'}

BUSINESS SUMMARY:
{(business_summary or '')[:800]}

UPCOMING EARNINGS:
{earnings_lines}

UPCOMING ECONOMIC EVENTS (high/medium impact only):
{econ_lines}

RECENT NEWS HEADLINES (last 14 days):
{news_lines}

Distill into 5-10 key forward catalysts per the system prompt. Be selective.
"""


def analyze_catalysts(
    c: CatalystSnapshot,
    sector: str | None = None,
    business_summary: str | None = None,
) -> ForwardCatalysts:
    user_msg = _format_input(c, sector, business_summary)
    result = chat_json(
        system=SYSTEM_PROMPT,
        user=user_msg,
        schema=ForwardCatalysts,
        temperature=0.2,
    )
    # Force symbol + horizon to match input exactly
    result.symbol = c.symbol
    result.horizon_days = c.horizon_days
    return result
