"""Convert recorded decisions into a portfolio equity curve.

Phase 2 default: long-only, equal-weight over Buy/Overweight names.
Phase 3 introduces a pluggable sizer — see ``tradingagents.portfolio``
for ``risk_aware_sizer`` with inverse-vol scaling and per-name/sector/gross
caps. ``build_equity_curve`` accepts any sizer that matches the ``Sizer``
protocol; the default remains equal-weight so legacy callers and tests
keep producing identical curves.

Price-fetching uses yfinance and respects ``data_cache_dir`` to avoid
re-downloading. All math is daily-close based with auto-adjustment for
splits/dividends (yfinance ``auto_adjust=True``).
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Callable, Iterable, Mapping, Optional, Sequence

import pandas as pd

from .strategy import Decision

logger = logging.getLogger(__name__)


# --- rating → weight mapping (default, equal-weight) ---------------------


_INCLUDE = {"Buy", "Overweight"}


def rating_to_weights(decisions_at_date: Iterable[Decision]) -> dict[str, float]:
    """Equal-weight every name rated Buy or Overweight. Others get 0.

    Kept for backwards compatibility — Phase 3's ``equal_weight_sizer``
    wraps this same logic with the broader Sizer protocol. If no name
    qualifies, returns an empty dict (portfolio sits in cash).
    """
    longs = [d.ticker for d in decisions_at_date if d.rating in _INCLUDE]
    if not longs:
        return {}
    w = 1.0 / len(longs)
    return {t: w for t in longs}


# --- price fetching --------------------------------------------------------


def fetch_prices(
    tickers: Sequence[str],
    start: str,
    end: str,
    benchmark: str | None = None,
) -> pd.DataFrame:
    """Download adjusted closes for ``tickers`` (and optional ``benchmark``).

    Returns a DataFrame indexed by date with one column per ticker. Uses
    yfinance's auto-adjusted prices so dividends and splits don't show up
    as fake jumps in the equity curve.

    Missing tickers (delisted, IPO after start, bad symbol) are silently
    dropped — the caller can compare ``df.columns`` to its requested
    universe to surface the gap.
    """
    import yfinance as yf

    symbols = list(dict.fromkeys([*tickers, benchmark] if benchmark else tickers))
    raw = yf.download(
        symbols,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
        group_by="ticker",
        threads=True,
    )
    if raw.empty:
        return pd.DataFrame()

    # yfinance returns a single-level frame when only one symbol is asked,
    # and a multi-level frame otherwise. Normalise to columns=ticker.
    if isinstance(raw.columns, pd.MultiIndex):
        closes = pd.DataFrame({
            sym: raw[sym]["Close"] for sym in symbols if sym in raw.columns.get_level_values(0)
        })
    else:
        closes = raw[["Close"]].rename(columns={"Close": symbols[0]})

    closes = closes.dropna(how="all").sort_index()
    closes.index = pd.to_datetime(closes.index).tz_localize(None)
    return closes


# --- equity curve construction --------------------------------------------


def build_equity_curve(
    decisions: Sequence[Decision],
    prices: pd.DataFrame,
    *,
    starting_capital: float = 1.0,
    sizer: Optional[Callable] = None,
    sectors: Optional[Mapping[str, str]] = None,
    sizing_config=None,
) -> pd.Series:
    """Compose the equity curve from recorded decisions and a price panel.

    For each rebalance date D in sorted order:
      1. Compute target weights via ``sizer(decisions_at_D, prices=..., as_of=D)``
      2. Hold those weights until the next rebalance D'
      3. Compute the daily-close portfolio return over [D, D']
      4. Compound into the equity series

    ``sizer`` defaults to the Phase 2 equal-weight rule (backwards
    compatible). Pass ``tradingagents.portfolio.risk_aware_sizer`` plus a
    ``SizingConfig`` and optional ``sectors`` map for vol-scaled + capped
    weights. The sizer is called once per rebalance date.

    The first rebalance date is the inception date — capital starts there
    so points before that are not part of the curve. If a target ticker
    has no price on the rebalance day, the next available trading day's
    close is used (yfinance trading-day reindexing).
    """
    if not decisions:
        return pd.Series(dtype=float)
    if prices.empty:
        return pd.Series(dtype=float)

    by_date: dict[str, list[Decision]] = defaultdict(list)
    for d in decisions:
        by_date[d.trade_date].append(d)
    rebalance_dates = sorted(by_date)

    trading_idx = prices.index
    snapped: list[pd.Timestamp] = []
    for rd in rebalance_dates:
        ts = pd.Timestamp(rd)
        candidate = trading_idx[trading_idx >= ts]
        if len(candidate) == 0:
            break
        snapped.append(candidate[0])

    if len(snapped) < 2:
        return pd.Series([starting_capital], index=snapped or [pd.Timestamp(rebalance_dates[0])])

    equity: list[float] = [starting_capital]
    equity_dates: list[pd.Timestamp] = [snapped[0]]

    # Two return-aggregation modes:
    #   - legacy (sizer is None): weights always sum to 1 (equal-weight),
    #     so a missing ticker is excluded from the average by normalising
    #     the period return by the realised weight sum.
    #   - explicit sizer: weights are absolute position fractions; gross
    #     may be < 1 (cash residual) or > 1 (leveraged). The return is the
    #     raw sum of w_i * r_i — no normalisation.
    normalise = sizer is None

    for i in range(len(snapped) - 1):
        start_day = snapped[i]
        end_day = snapped[i + 1]
        if sizer is None:
            weights = rating_to_weights(by_date[rebalance_dates[i]])
        else:
            weights = sizer(
                by_date[rebalance_dates[i]],
                prices=prices,
                sectors=sectors,
                as_of=start_day,
                config=sizing_config,
            )

        if not weights:
            equity.append(equity[-1])
            equity_dates.append(end_day)
            continue

        period_ret = 0.0
        active_weight = 0.0
        for tkr, w in weights.items():
            if tkr not in prices.columns:
                continue
            series = prices[tkr]
            p_start = series.loc[start_day] if start_day in series.index else None
            p_end = series.loc[end_day] if end_day in series.index else None
            if p_start is None or p_end is None or pd.isna(p_start) or pd.isna(p_end):
                continue
            period_ret += w * (p_end / p_start - 1)
            active_weight += abs(w)

        if active_weight == 0:
            equity.append(equity[-1])
        elif normalise:
            equity.append(equity[-1] * (1 + period_ret / active_weight))
        else:
            equity.append(equity[-1] * (1 + period_ret))
        equity_dates.append(end_day)

    return pd.Series(equity, index=equity_dates, name="equity")


def benchmark_curve(
    prices: pd.DataFrame,
    benchmark: str,
    equity_index: pd.Index,
    *,
    starting_capital: float = 1.0,
) -> pd.Series:
    """Normalise a benchmark price series to the equity curve's anchors.

    Returns a Series on the same index as ``equity_index`` so the two can
    be plotted or differenced directly.
    """
    if benchmark not in prices.columns:
        return pd.Series(dtype=float)
    bench = prices[benchmark].reindex(equity_index, method="ffill")
    if bench.empty or pd.isna(bench.iloc[0]):
        return pd.Series(dtype=float)
    return starting_capital * (bench / bench.iloc[0])
