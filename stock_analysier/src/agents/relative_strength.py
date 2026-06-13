"""Relative Strength Agent — pure deterministic ticker-vs-benchmark return ratios.

Compares the ticker's cumulative return over each configured window against the
sector ETF and market index returns over the same window. RS ratio = ticker_ret /
benchmark_ret. RS rank vs other names is computed by the orchestrator (not here)
since it requires the active watchlist.

No LLM. Receives only ticker symbol + market + sector + config. Blinded from
the fundamental side.
"""
from __future__ import annotations

import logging
from typing import Literal, Optional

from src.data.benchmarks import fetch_returns, get_market_index, get_sector_etf
from src.models.schemas import RelativeStrengthAssessment, RelativeStrengthCfg

log = logging.getLogger(__name__)


def _safe_ratio(ticker_pct: Optional[float], bench_pct: Optional[float]) -> Optional[float]:
    """Compute (1+r_ticker)/(1+r_bench). None if either is missing."""
    if ticker_pct is None or bench_pct is None:
        return None
    return round((1.0 + ticker_pct / 100.0) / (1.0 + bench_pct / 100.0), 3)


def _classify_signal(ratios: list[Optional[float]]) -> Literal[
    "strong_leader", "leader", "neutral", "laggard", "weak_laggard"
]:
    """Average the available RS ratios and band the result."""
    present = [r for r in ratios if r is not None]
    if not present:
        return "neutral"
    avg = sum(present) / len(present)
    if avg >= 1.20:
        return "strong_leader"
    if avg >= 1.05:
        return "leader"
    if avg >= 0.95:
        return "neutral"
    if avg >= 0.80:
        return "laggard"
    return "weak_laggard"


def assess_relative_strength(
    symbol: str,
    market: str,
    sector: Optional[str],
    ticker_returns: dict[int, Optional[float]],
    cfg: RelativeStrengthCfg,
) -> RelativeStrengthAssessment:
    """Compute RS ratios vs the configured sector ETF + market index.

    `ticker_returns` is { window_days: cumulative_pct } for the ticker — pre-computed
    by the orchestrator from existing price history so we don't double-fetch.
    """
    sector_etf = get_sector_etf(market, sector)
    market_index = get_market_index(market)

    sector_returns: dict[int, Optional[float]] = {}
    if sector_etf:
        try:
            sector_returns = fetch_returns(sector_etf, cfg.windows_days)
        except Exception as e:
            log.warning("RS: failed to fetch sector returns for %s: %s", sector_etf, e)

    index_returns: dict[int, Optional[float]] = {}
    if market_index:
        try:
            index_returns = fetch_returns(market_index, cfg.windows_days)
        except Exception as e:
            log.warning("RS: failed to fetch index returns for %s: %s", market_index, e)

    # Convention: 90d window first, 365d second (the schema only has these two slots)
    w90 = 90 if 90 in cfg.windows_days else (cfg.windows_days[0] if cfg.windows_days else None)
    w365 = 365 if 365 in cfg.windows_days else (cfg.windows_days[-1] if cfg.windows_days else None)

    vs_sector_90 = _safe_ratio(ticker_returns.get(w90), sector_returns.get(w90)) if w90 else None
    vs_sector_365 = _safe_ratio(ticker_returns.get(w365), sector_returns.get(w365)) if w365 else None
    vs_index_90 = _safe_ratio(ticker_returns.get(w90), index_returns.get(w90)) if w90 else None
    vs_index_365 = _safe_ratio(ticker_returns.get(w365), index_returns.get(w365)) if w365 else None

    signal = _classify_signal([vs_sector_90, vs_sector_365, vs_index_90, vs_index_365])

    return RelativeStrengthAssessment(
        vs_sector_etf_90d=vs_sector_90,
        vs_sector_etf_365d=vs_sector_365,
        vs_index_90d=vs_index_90,
        vs_index_365d=vs_index_365,
        rs_rank_in_universe=None,  # populated by the orchestrator if a universe is available
        signal=signal,
        benchmark_sector_etf=sector_etf,
        benchmark_index=market_index,
    )
