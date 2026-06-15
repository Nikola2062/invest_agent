"""Cross-sectional relative-strength ranking + coarse market regime (Phase 3).

TradingAgents decides each name in isolation; nothing in the agent stack asks
"of all the things I *could* hold, which are actually leading the market right
now?". This module supplies that missing cross-sectional view: it ranks a
candidate set (held + watchlist) by relative strength versus a benchmark so
capital flows to the strongest opportunities, and it distills a coarse
**market regime** label from breadth that can feed the position_overlay danger
gate as an optional ``regime``.

Design mirrors ``position_overlay`` and ``sizing``: a **PURE core** (no network,
no LLM — every input is a value or a DataFrame, so it is unit-testable offline)
plus a **thin network wrapper** that fetches prices via yfinance and never
raises.

Relative strength = the asset's trailing return *minus* the benchmark's
trailing return, blended across several lookback windows (shorter windows
weighted a bit more so the score is responsive but not whippy).

The regime label ``"Risk-Off Defensive"`` is chosen to match
``OverlayConfig.risk_off_regimes`` in ``position_overlay.py`` exactly, so a
risk-off reading from this module trips the overlay's danger gate.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

# Default lookback windows (trading days) and their blend weights. Shorter
# windows are weighted a bit more so the score reacts to fresh leadership while
# still respecting the multi-month trend.
DEFAULT_LOOKBACKS = (21, 63, 126)
_DEFAULT_WEIGHTS = (0.5, 0.3, 0.2)

# Coarse-regime thresholds (see ``coarse_regime``).
RISK_ON_BREADTH = 0.60      # > this fraction outperforming + benchmark up -> Risk-On
RISK_OFF_BREADTH = 0.40     # < this fraction outperforming -> Risk-Off Defensive
BENCH_DOWN_MATERIAL = -0.03  # benchmark trailing return below this -> materially down

# Must match OverlayConfig.risk_off_regimes in position_overlay.py exactly.
REGIME_RISK_ON = "Risk-On"
REGIME_NEUTRAL = "Neutral"
REGIME_RISK_OFF = "Risk-Off Defensive"


# --- pure helpers ---------------------------------------------------------


def _trailing_return(prices: pd.Series, lookback: int) -> Optional[float]:
    """Simple trailing return over ``lookback`` bars, or None if too short.

    Uses the last value vs the value ``lookback`` bars earlier. NaNs are
    dropped first so a sparsely-quoted series still produces a usable number.
    """
    s = prices.dropna()
    if len(s) <= lookback:
        return None
    start = s.iloc[-(lookback + 1)]
    end = s.iloc[-1]
    if start == 0 or pd.isna(start) or pd.isna(end):
        return None
    return float(end / start - 1.0)


# --- pure core ------------------------------------------------------------


def relative_strength_score(
    prices: pd.Series,
    benchmark: pd.Series,
    lookbacks=DEFAULT_LOOKBACKS,
    weights=_DEFAULT_WEIGHTS,
) -> float:
    """Weighted blend of (asset trailing return − benchmark trailing return).

    For each lookback window the asset's trailing return minus the benchmark's
    trailing return is computed, then the windows are blended with ``weights``
    (renormalised over whichever windows are usable).

    Lookbacks longer than the available history are skipped. If *no* lookback
    is usable (series too short) the function returns ``0.0`` — a neutral score
    that ranks such a name in the middle rather than dropping it or poisoning
    the sort with NaN.
    """
    contribs: list[tuple[float, float]] = []  # (weight, relative_return)
    for lb, w in zip(lookbacks, weights):
        a = _trailing_return(prices, lb)
        b = _trailing_return(benchmark, lb)
        if a is None or b is None:
            continue
        contribs.append((w, a - b))
    if not contribs:
        return 0.0
    total_w = sum(w for w, _ in contribs)
    if total_w == 0:
        return 0.0
    return float(sum(w * r for w, r in contribs) / total_w)


def rank(
    prices: pd.DataFrame,
    benchmark_col: str,
    tickers: Optional[list[str]] = None,
    lookbacks=DEFAULT_LOOKBACKS,
    weights=_DEFAULT_WEIGHTS,
) -> list[dict]:
    """Rank ``tickers`` by relative strength vs ``benchmark_col``, best first.

    ``prices`` is a DataFrame of Close prices, one column per ticker, and it
    MUST include ``benchmark_col``. ``tickers`` defaults to every column except
    the benchmark; if given, only those (that are present) are ranked. The
    benchmark itself is always excluded from the output.

    Returns a list sorted best-first::

        {"symbol", "rs_score", "rank", "trailing_returns": {lookback: pct}}

    ``trailing_returns`` maps each usable lookback to the asset's own trailing
    return in **percent** (e.g. 12.3 for +12.3%); unusable lookbacks are
    omitted. Ties break by symbol for determinism.
    """
    if benchmark_col not in prices.columns:
        raise KeyError(f"benchmark column {benchmark_col!r} not in prices")

    bench = prices[benchmark_col]
    if tickers is None:
        candidates = [c for c in prices.columns if c != benchmark_col]
    else:
        candidates = [t for t in tickers if t != benchmark_col and t in prices.columns]

    rows: list[dict] = []
    for sym in candidates:
        series = prices[sym]
        score = relative_strength_score(series, bench, lookbacks, weights)
        trailing: dict[int, float] = {}
        for lb in lookbacks:
            r = _trailing_return(series, lb)
            if r is not None:
                trailing[lb] = round(r * 100.0, 2)
        rows.append({"symbol": sym, "rs_score": score, "trailing_returns": trailing})

    rows.sort(key=lambda d: (-d["rs_score"], d["symbol"]))
    for i, row in enumerate(rows, start=1):
        row["rank"] = i
    return rows


def coarse_regime(
    prices: pd.DataFrame,
    benchmark_col: str,
    lookback: int = 63,
) -> str:
    """Distill a coarse market regime from breadth + benchmark trend.

    "Breadth" = the fraction of (non-benchmark) tickers whose trailing return
    over ``lookback`` beats the benchmark's. Combined with the sign/strength of
    the benchmark's own trailing return:

      * **Risk-On**            — breadth > 60% AND benchmark trailing return > 0.
      * **Risk-Off Defensive** — breadth < 40% OR benchmark down materially
                                  (trailing return below ``BENCH_DOWN_MATERIAL``,
                                  default −3%).
      * **Neutral**            — everything in between (including an empty /
                                  unusable panel).

    The ``"Risk-Off Defensive"`` string matches
    ``OverlayConfig.risk_off_regimes`` exactly, so it trips the overlay's
    danger gate when passed through as ``regime``.
    """
    if benchmark_col not in prices.columns:
        raise KeyError(f"benchmark column {benchmark_col!r} not in prices")

    bench = prices[benchmark_col]
    bench_ret = _trailing_return(bench, lookback)
    candidates = [c for c in prices.columns if c != benchmark_col]

    if bench_ret is None or not candidates:
        return REGIME_NEUTRAL

    outperformers = 0
    counted = 0
    for sym in candidates:
        r = _trailing_return(prices[sym], lookback)
        if r is None:
            continue
        counted += 1
        if r > bench_ret:
            outperformers += 1

    breadth = (outperformers / counted) if counted else 0.0

    if breadth < RISK_OFF_BREADTH or bench_ret < BENCH_DOWN_MATERIAL:
        return REGIME_RISK_OFF
    if breadth > RISK_ON_BREADTH and bench_ret > 0:
        return REGIME_RISK_ON
    return REGIME_NEUTRAL


# --- thin network wrapper (never raises) ----------------------------------


def fetch_prices(
    tickers: list[str],
    benchmark: str = "SPY",
    period: str = "6mo",
) -> pd.DataFrame:
    """Download Close prices for ``tickers`` + ``benchmark`` via yfinance.

    Returns a DataFrame indexed by date with one column per symbol (the
    benchmark column is named ``benchmark``). Auto-adjusted closes are used so
    dividends/splits don't masquerade as relative-strength jumps. Any failure
    (network, bad symbols, empty result) returns an **empty DataFrame** and
    never raises — callers can treat empty as "no signal / Neutral".
    """
    try:
        import yfinance as yf

        symbols = list(dict.fromkeys([*tickers, benchmark]))
        raw = yf.download(
            symbols,
            period=period,
            progress=False,
            auto_adjust=True,
            group_by="ticker",
            threads=True,
        )
        if raw is None or raw.empty:
            return pd.DataFrame()

        # yfinance returns a single-level frame for one symbol, multi-level
        # otherwise. Normalise to columns=symbol of Close prices.
        if isinstance(raw.columns, pd.MultiIndex):
            closes = pd.DataFrame({
                sym: raw[sym]["Close"]
                for sym in symbols
                if sym in raw.columns.get_level_values(0)
            })
        else:
            closes = raw[["Close"]].rename(columns={"Close": symbols[0]})

        closes = closes.dropna(how="all").sort_index()
        if closes.empty:
            return pd.DataFrame()
        closes.index = pd.to_datetime(closes.index).tz_localize(None)
        return closes
    except Exception:
        return pd.DataFrame()


def fetch_and_rank(tickers: list[str], benchmark: str = "SPY") -> list[dict]:
    """Convenience: fetch prices then ``rank`` vs the benchmark.

    Returns ``[]`` if prices could not be fetched or the benchmark is missing.
    """
    prices = fetch_prices(tickers, benchmark=benchmark)
    if prices.empty or benchmark not in prices.columns:
        return []
    return rank(prices, benchmark_col=benchmark, tickers=tickers)


def fetch_regime(tickers: list[str], benchmark: str = "SPY") -> str:
    """Convenience: fetch prices then derive ``coarse_regime``.

    Returns ``"Neutral"`` (the safe, non-gating default) if prices could not be
    fetched or the benchmark is missing.
    """
    prices = fetch_prices(tickers, benchmark=benchmark)
    if prices.empty or benchmark not in prices.columns:
        return REGIME_NEUTRAL
    return coarse_regime(prices, benchmark_col=benchmark)
