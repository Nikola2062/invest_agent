"""Performance metrics for an equity curve.

Pure pandas — no quantstats dependency. The functions accept either an
equity series or a returns series; ``returns_from_equity`` converts
between them. All annualisation uses an explicit ``periods_per_year``
argument so the same code works for daily, weekly, or monthly curves.

Conventions:
  - Returns are arithmetic (``r = P_t / P_{t-1} - 1``), not log.
  - Sharpe and Sortino use risk-free rate = 0 by default. Pass ``rf``
    annualised if you need to subtract a non-zero benchmark.
  - Max drawdown is reported as a negative number (e.g. ``-0.18`` for -18%).
  - Information ratio is computed against an explicit benchmark series,
    not against ``rf``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd


# --- conversions ----------------------------------------------------------


def returns_from_equity(equity: pd.Series) -> pd.Series:
    """Arithmetic period returns from an equity curve."""
    return equity.pct_change().dropna()


def _infer_periods_per_year(index: pd.Index) -> float:
    """Estimate annualisation factor from index spacing.

    Backtests at this level run weekly or monthly. We infer from the median
    interval between rebalances so the metrics function doesn't need to be
    told the frequency. Daily-spaced data is treated as 252 trading days.
    """
    if len(index) < 2:
        return 252.0
    deltas = pd.Series(index).diff().dropna().dt.days
    if deltas.empty:
        return 252.0
    median_days = float(deltas.median())
    if median_days <= 1.5:
        return 252.0
    if median_days <= 8:
        return 52.0
    if median_days <= 35:
        return 12.0
    if median_days <= 100:
        return 4.0
    return 1.0


# --- core metrics ---------------------------------------------------------


def total_return(equity: pd.Series) -> float:
    if equity.empty:
        return float("nan")
    return float(equity.iloc[-1] / equity.iloc[0] - 1)


def cagr(equity: pd.Series, periods_per_year: float | None = None) -> float:
    """Compound annual growth rate."""
    if equity.empty or len(equity) < 2:
        return float("nan")
    ppy = periods_per_year or _infer_periods_per_year(equity.index)
    n_periods = len(equity) - 1
    years = n_periods / ppy
    if years <= 0:
        return float("nan")
    return float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1)


def annualised_volatility(returns: pd.Series, periods_per_year: float | None = None) -> float:
    if returns.empty:
        return float("nan")
    ppy = periods_per_year or _infer_periods_per_year(returns.index)
    return float(returns.std(ddof=1) * np.sqrt(ppy))


def sharpe_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: float | None = None,
) -> float:
    if returns.empty:
        return float("nan")
    ppy = periods_per_year or _infer_periods_per_year(returns.index)
    excess = returns - rf / ppy
    sd = excess.std(ddof=1)
    if sd == 0 or pd.isna(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(ppy))


def sortino_ratio(
    returns: pd.Series,
    rf: float = 0.0,
    periods_per_year: float | None = None,
) -> float:
    if returns.empty:
        return float("nan")
    ppy = periods_per_year or _infer_periods_per_year(returns.index)
    excess = returns - rf / ppy
    downside = excess.where(excess < 0, 0.0)
    dd = np.sqrt((downside ** 2).mean())
    if dd == 0 or pd.isna(dd):
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(ppy))


def max_drawdown(equity: pd.Series) -> float:
    """Return max drawdown as a negative fraction (e.g. -0.18 = -18%)."""
    if equity.empty:
        return float("nan")
    running_max = equity.cummax()
    drawdowns = equity / running_max - 1
    return float(drawdowns.min())


def calmar_ratio(equity: pd.Series, periods_per_year: float | None = None) -> float:
    """CAGR / |max drawdown|."""
    mdd = max_drawdown(equity)
    if pd.isna(mdd) or mdd == 0:
        return float("nan")
    return float(cagr(equity, periods_per_year) / abs(mdd))


def hit_rate(returns: pd.Series) -> float:
    """Fraction of periods with strictly positive return."""
    if returns.empty:
        return float("nan")
    return float((returns > 0).mean())


def alpha_vs_benchmark(equity: pd.Series, benchmark: pd.Series) -> float:
    """Total-return alpha vs benchmark over the same window."""
    if equity.empty or benchmark.empty:
        return float("nan")
    return total_return(equity) - total_return(benchmark)


def information_ratio(
    returns: pd.Series,
    benchmark_returns: pd.Series,
    periods_per_year: float | None = None,
) -> float:
    """Annualised mean active return / tracking error.

    Active return = portfolio return - benchmark return, on aligned dates.
    The two series are inner-joined on index, then the std of the
    difference is the tracking error.
    """
    aligned = pd.concat([returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.empty:
        return float("nan")
    active = aligned.iloc[:, 0] - aligned.iloc[:, 1]
    sd = active.std(ddof=1)
    if sd == 0 or pd.isna(sd):
        return float("nan")
    ppy = periods_per_year or _infer_periods_per_year(active.index)
    return float(active.mean() / sd * np.sqrt(ppy))


# --- summary --------------------------------------------------------------


@dataclass
class Summary:
    n_periods: int
    total_return: float
    cagr: float
    volatility: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    hit_rate: float
    alpha_vs_benchmark: float | None = None
    information_ratio: float | None = None
    benchmark_total_return: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def summarise(
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    *,
    rf: float = 0.0,
    periods_per_year: float | None = None,
) -> Summary:
    """One-call summary stats for an equity curve.

    ``benchmark`` (if provided) should already be on a comparable scale —
    use ``portfolio.benchmark_curve`` to normalise it to the same starting
    capital and index as ``equity``.
    """
    rets = returns_from_equity(equity)
    out = Summary(
        n_periods=len(rets),
        total_return=total_return(equity),
        cagr=cagr(equity, periods_per_year),
        volatility=annualised_volatility(rets, periods_per_year),
        sharpe=sharpe_ratio(rets, rf=rf, periods_per_year=periods_per_year),
        sortino=sortino_ratio(rets, rf=rf, periods_per_year=periods_per_year),
        max_drawdown=max_drawdown(equity),
        calmar=calmar_ratio(equity, periods_per_year),
        hit_rate=hit_rate(rets),
    )
    if benchmark is not None and not benchmark.empty:
        bench_rets = returns_from_equity(benchmark)
        out.alpha_vs_benchmark = alpha_vs_benchmark(equity, benchmark)
        out.information_ratio = information_ratio(rets, bench_rets, periods_per_year)
        out.benchmark_total_return = total_return(benchmark)
    return out
