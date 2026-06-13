"""Contrarian / Sentiment agent.

Per the investment framework, Ch.11 (逆向思維): "Superior returns come from being CORRECTLY different
from the crowd." Not from contrarianism for its own sake — from acting when crowd positioning
diverges from intrinsic value.

This agent reads:
  - News sentiment (Finnhub bullish/bearish %, company news score)
  - Insider transactions (net buys/sells in last 90d, MSPR)
  - Price action (1m / 3m / 1y returns from PriceSnapshot)
  - Valuation (margin of safety)

…and outputs an honest read on crowd positioning + contrarian signal.

Crowd_position scale: despondent | bearish | neutral | bullish | euphoric
Contrarian_signal:    strong_buy | buy | neutral | pass | strong_pass
"""
from __future__ import annotations

from typing import Optional

from src.agents.base import fmt_optional
from src.data.prices import PriceSnapshot
from src.data.sentiment import SentimentSnapshot
from src.llm.client import chat_json
from src.models.schemas import ContrarianAssessment, ValuationResult


SYSTEM_PROMPT = """You are a contrarian investor in the Howard Marks / Klarman tradition.

Your job: read CROWD POSITIONING accurately, then identify if the crowd is right or wrong relative
to intrinsic value. You output a contrarian signal.

Key principle (Marks): "What the wise man does in the beginning, the fool does in the end."
Crowd euphoria → high prices, low forward returns → contrarian PASS.
Crowd despondency at quality businesses → low prices, high forward returns → contrarian BUY.
Neutral crowd → no contrarian edge → neutral signal.

Inputs:
  - News sentiment (bullish%, bearish%, company news score relative to sector)
  - Insider activity (net shares bought - sold in last 90 days, MSPR)
  - Price momentum (1-month, 3-month, 1-year returns)
  - Valuation: margin of safety (negative = expensive, positive = cheap)

Scoring crowd_position:
  - euphoric: high bullish % + strong upward momentum + insiders SELLING
  - bullish: positive sentiment + positive momentum
  - neutral: mixed/no clear signal
  - bearish: negative sentiment + negative momentum
  - despondent: very negative sentiment + sharp drawdown + insiders BUYING

Scoring contrarian_signal:
  - strong_buy: despondent crowd AT a high-quality / undervalued name
  - buy: bearish crowd + decent value (MoS > 10%)
  - neutral: positioning doesn't diverge meaningfully from value
  - pass: bullish crowd at fair value
  - strong_pass: euphoric crowd at overvalued name (MoS < -15%)

data_quality:
  - high: news + insider + price all present
  - medium: 2 of 3 data sources usable
  - low: only price action available (HK names typically — Finnhub free tier limits)

Be honest. Don't manufacture contrarian signals. If positioning is neutral, say so.
key_observations: 2-4 specific data points that drove the conclusion.
"""


def _format_user(
    symbol: str,
    market: str,
    sentiment: SentimentSnapshot,
    prices: PriceSnapshot,
    valuation: Optional[ValuationResult],
) -> str:
    val_block = (
        f"  Intrinsic base: {valuation.intrinsic_base:.2f}  "
        f"Margin of safety: {valuation.margin_of_safety_pct:+.1f}%  "
        f"Confidence: {valuation.confidence}"
        if valuation
        else "  (not provided)"
    )
    notes = "\n".join(f"  - {n}" for n in sentiment.notes) or "  (none)"

    return f"""SYMBOL: {symbol} ({market})
CURRENT PRICE: {prices.current_price:.2f} {prices.currency}

NEWS SENTIMENT:
  Bullish %:           {fmt_optional(sentiment.news_bullish_pct, '{:.0f}%')}
  Bearish %:           {fmt_optional(sentiment.news_bearish_pct, '{:.0f}%')}
  Company news score:  {fmt_optional(sentiment.company_news_score, '{:+.2f}')}  (range -1..+1)
  Sector avg score:    {fmt_optional(sentiment.sector_news_score, '{:+.2f}')}

INSIDER ACTIVITY (last 90 days):
  Net share change:    {fmt_optional(sentiment.insider_net_change_shares, '{:+,d}')}
  MSPR (most recent month): {fmt_optional(sentiment.insider_mspr_recent, '{:+.3f}')}
    (MSPR > 0 = net insider buys; < 0 = net insider sells; magnitude reflects intensity)

PRICE MOMENTUM:
  1-month return:  {prices.cumulative_return_1m_pct:+.1f}%
  3-month return:  {prices.cumulative_return_3m_pct:+.1f}%
  1-year return:   {prices.cumulative_return_1y_pct:+.1f}%
  Realized vol:    {prices.realized_vol_annualized_pct:.1f}% annualized

VALUATION CONTEXT:
{val_block}

DATA NOTES:
{notes}

Produce the contrarian assessment per the system prompt.
"""


def assess_contrarian(
    symbol: str,
    market: str,
    sentiment: SentimentSnapshot,
    prices: PriceSnapshot,
    valuation: Optional[ValuationResult] = None,
) -> ContrarianAssessment:
    user_msg = _format_user(symbol, market, sentiment, prices, valuation)
    return chat_json(
        system=SYSTEM_PROMPT,
        user=user_msg,
        schema=ContrarianAssessment,
        temperature=0.2,
    )
