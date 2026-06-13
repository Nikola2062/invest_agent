"""Portfolio Fit agent — cross-position correlation + risk-source concentration.

Reads the current holdings book, plus any watchlist tickers actively flagged BUY_NOW
in their AnalysisResults. Computes pairwise 90d correlation matrix. LLM identifies
clusters and concentration risks ("if all your positions express the same one bet,
you don't have a diversified book — you have leverage on that bet").

Per the investment framework, Ch.8 (分散化): "diversification of risk sources, not stock count."
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from src.config.loader import load_portfolio
from src.data.fx import convert as fx_convert
from src.llm.client import chat_json
from src.models.schemas import (
    AnalysisResult,
    PortfolioFitReport,
)

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a portfolio risk officer analyzing concentration risk across a book.

True diversification (per Buffett / Yale / Klarman framework) means owning DIFFERENT risk sources.
Owning 10 cloud-software stocks is not diversification — it is leverage on one factor.

You receive:
  - List of positions with sector, value_usd, current weight
  - Pairwise 90-day correlation matrix between all positions

Your job:
  - Identify CLUSTERS — groups of positions with average pairwise correlation > 0.5 that share an underlying risk source
  - For each cluster, name the COMMON RISK SOURCE (e.g., "global tech sentiment / NASDAQ beta", "China internet regulation / consumer", "USD-strength carry", "long-duration assets / rate-cut beta")
  - Estimate concentration: % of total book value in the cluster
  - Severity:
      low    = cluster < 30% of book
      medium = cluster 30-60% of book
      high   = cluster > 60% of book

Output (PortfolioFitReport):
  - clusters: list of CorrelationCluster
  - concentration_warnings: 1-2 sentences each, FOCUSED on the highest-severity clusters
  - diversification_recommendations: 2-4 concrete asset-class / sector / geographic additions that would meaningfully reduce the dominant risk concentration (e.g., "add a defensive staples or healthcare allocation", "add a USD-bond sleeve", "add a non-US developed market exposure outside HK")
  - diversification_score: 0-10. Honest grading:
      9-10: 3+ uncorrelated risk sources, no single cluster > 30%
      7-8:  2 distinct risk sources, dominant cluster 30-50%
      5-6:  one dominant cluster 50-65%, some diversification
      3-4:  one dominant cluster > 65%, weak diversification
      0-2:  effectively a single-factor portfolio
  - summary: 1-2 sentences. Direct and honest.

Discipline:
  - Don't fabricate diversification that isn't there.
  - Don't recommend asset classes you can't justify.
  - If the book is small (1-2 positions), say so — diversification is structurally impossible at that scale, and the recommendation is to size positions accordingly.
"""


@dataclass
class _Position:
    symbol: str
    market: str
    sector: Optional[str]
    value_usd: float
    weight_pct: float
    current_price: float
    currency: str


def _to_usd(value: float, currency: str) -> float:
    return fx_convert(value, currency, "USD")


def _build_position_list(results: list[AnalysisResult]) -> list[_Position]:
    portfolio = load_portfolio()
    held_symbols = {h.symbol for h in portfolio.holdings}

    positions: list[_Position] = []
    # Held: use actual share counts
    for h in portfolio.holdings:
        match = next((r for r in results if r.symbol == h.symbol), None)
        price = match.current_price if match else h.cost_basis_per_share
        value_local = h.shares * price
        positions.append(_Position(
            symbol=h.symbol, market=h.market,
            sector=match.fundamental.moat_strength if False else None,  # placeholder; we'll fill below
            value_usd=_to_usd(value_local, h.currency),
            weight_pct=0.0,  # filled later
            current_price=price,
            currency=h.currency,
        ))

    # Watchlist actionable: only those flagged BUY_NOW — we add them as candidate positions
    # at a hypothetical 5% notional weight (so they don't dominate the diversification math)
    book_value_usd = sum(p.value_usd for p in positions) or 1.0
    hypothetical_size_usd = book_value_usd * 0.05
    for r in results:
        if r.symbol in held_symbols:
            continue
        if r.if_not_held.recommendation != "BUY_NOW":
            continue
        positions.append(_Position(
            symbol=r.symbol, market=r.market, sector=None,
            value_usd=hypothetical_size_usd,
            weight_pct=0.0,
            current_price=r.current_price,
            currency=r.currency,
        ))

    # Fill sectors from results when available
    for p in positions:
        match = next((r for r in results if r.symbol == p.symbol), None)
        if match is not None:
            # We don't have sector on the result directly; reach into fundamentals snapshot? Not stored.
            # Use moat strength as a soft proxy is wrong — instead, just leave None and let LLM rely on symbol+market.
            pass

    total = sum(p.value_usd for p in positions) or 1.0
    for p in positions:
        p.weight_pct = p.value_usd / total * 100

    return positions


def _correlation_matrix(symbols: list[str], days: int = 90) -> Optional[pd.DataFrame]:
    if len(symbols) < 2:
        return None
    period = f"{max(days + 15, 120)}d"
    try:
        df = yf.download(
            symbols, period=period, auto_adjust=True, progress=False,
            group_by="ticker", threads=True,
        )
    except Exception as e:
        log.warning("yf.download for portfolio fit failed: %s", e)
        return None

    closes: dict[str, pd.Series] = {}
    for s in symbols:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                series = df[s]["Close"].dropna()
            else:
                series = df["Close"].dropna()
            if not series.empty:
                closes[s] = series
        except Exception:
            continue

    if len(closes) < 2:
        return None

    rets = pd.DataFrame({k: np.log(v / v.shift(1)) for k, v in closes.items()}).dropna().tail(days)
    if len(rets) < 20:
        return None
    return rets.corr()


def _format_user(positions: list[_Position], corr: Optional[pd.DataFrame]) -> str:
    book = sum(p.value_usd for p in positions)
    pos_lines = "\n".join(
        f"  - {p.symbol} ({p.market}): ${p.value_usd:,.0f} ({p.weight_pct:.1f}% of book), "
        f"current ${p.current_price:.2f} {p.currency}"
        for p in positions
    )

    if corr is None:
        corr_block = "  (insufficient data — fewer than 2 positions with price history)"
    else:
        # Pretty-print the matrix
        corr_block = corr.round(2).to_string()

    return f"""TOTAL BOOK VALUE: ${book:,.0f}

POSITIONS:
{pos_lines}

90-DAY DAILY RETURN CORRELATION MATRIX:
{corr_block}

Analyze concentration per the system prompt. Be honest about diversification reality at this book size.
"""


def analyze_portfolio_fit(results: list[AnalysisResult]) -> Optional[PortfolioFitReport]:
    """Run the Portfolio Fit agent across all current results.

    Returns None if there's nothing to analyze (no holdings + no BUY_NOWs).
    """
    positions = _build_position_list(results)
    if len(positions) < 1:
        return None

    symbols = [p.symbol for p in positions]
    corr = _correlation_matrix(symbols, days=90) if len(symbols) >= 2 else None

    user_msg = _format_user(positions, corr)
    report = chat_json(
        system=SYSTEM_PROMPT,
        user=user_msg,
        schema=PortfolioFitReport,
        temperature=0.25,
    )
    report.total_positions = len(positions)
    report.total_book_value_usd = sum(p.value_usd for p in positions)
    return report
