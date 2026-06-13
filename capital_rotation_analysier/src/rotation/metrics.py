"""Per-asset and cross-sectional metrics, per the project docs §3.1.

Pure functions. Inputs are pandas DataFrames indexed by date with one column per symbol.
Outputs are pandas DataFrames with the same index/columns where applicable.

All return calculations are LOG returns (time-additive, symmetric, ~normal at daily).
Arithmetic returns are only for the final report layer.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


# ---------- core return / vol primitives ----------

def log_returns(close: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """ln(C_t / C_{t-horizon}). Returns NaN for the first `horizon` rows."""
    return np.log(close).diff(horizon)


def realized_vol(close: pd.DataFrame, window: int = 30, min_obs: int | None = None) -> pd.DataFrame:
    """Annualized realized vol of daily log returns."""
    if min_obs is None:
        min_obs = max(1, int(window * 0.8))
    r = log_returns(close, 1)
    return r.rolling(window=window, min_periods=min_obs).std() * np.sqrt(252)


def vol_of_vol_ratio(close: pd.DataFrame, fast: int = 10, slow: int = 60) -> pd.DataFrame:
    """σ_fast / σ_slow. A value > 1.5 flags a vol regime break."""
    return realized_vol(close, fast) / realized_vol(close, slow)


# ---------- volume metrics ----------

def relative_volume(volume: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    """ln(V_t / median(V_{t-window:t-1})). Robust to spikes via median."""
    # shift(1) so today's volume isn't in its own normalization window
    med = volume.shift(1).rolling(window=window, min_periods=int(window * 0.8)).median()
    with np.errstate(divide="ignore", invalid="ignore"):
        rv = np.log(volume / med)
    return rv.replace([np.inf, -np.inf], np.nan)


def volume_zscore(volume: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Z-score on LOG volume. Raw volume is log-normal; z-scoring raw V is fat-tailed garbage."""
    log_v = np.log(volume.where(volume > 0))
    mu = log_v.shift(1).rolling(window=window, min_periods=int(window * 0.8)).mean()
    sd = log_v.shift(1).rolling(window=window, min_periods=int(window * 0.8)).std()
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (log_v - mu) / sd
    return z.replace([np.inf, -np.inf], np.nan)


# ---------- cross-sectional relative strength ----------

def blended_return(
    close: pd.DataFrame,
    weights: dict[int, float] = None,
) -> pd.DataFrame:
    """Blended-horizon return: 0.4·r_m + 0.3·r_w + 0.2·r_q + 0.1·r_d (defaults from §3.1)."""
    weights = weights or {21: 0.4, 5: 0.3, 63: 0.2, 1: 0.1}
    out = None
    for h, w in weights.items():
        r = log_returns(close, h) * w
        out = r if out is None else out.add(r, fill_value=0)
    return out


def rs_rank(close: pd.DataFrame) -> pd.DataFrame:
    """IBD-style 1-99 cross-sectional rank of blended return.

    Honest theater with N≈17 — adjacent ranks are within noise. We report it for
    readability; we do NOT trade off rank changes alone. The underlying signal is
    blended_return itself.
    """
    br = blended_return(close)
    ranks = br.rank(axis=1, pct=True)  # 0..1
    return (ranks * 98 + 1).round().astype("Int64")  # 1..99


def rank_change(rs: pd.DataFrame, k: int) -> pd.DataFrame:
    """RS_t − RS_{t-k}."""
    return rs - rs.shift(k)


# ---------- robust z-scoring (the §3.2 common pattern) ----------

def robust_z(
    x: pd.Series | pd.DataFrame,
    window: int = 252,
    min_obs: int = 126,
    clip: float = 3.0,
) -> pd.Series | pd.DataFrame:
    """(x_t − median_W) / (1.4826 · MAD_W) over a trailing window W, excluding the
    current observation from W to avoid look-ahead bias. Clipped to [-clip, +clip].

    Implementation note: computing median and MAD in a single rolling.apply pass
    (instead of chained .rolling().median() ops) preserves the effective sample
    size — chaining two rolling windows compounds the warm-up period and drops
    valid observations needlessly.
    """
    def _z(arr: np.ndarray) -> float:
        prior = arr[:-1]
        prior = prior[~np.isnan(prior)]
        if prior.size < min_obs:
            return np.nan
        m = float(np.median(prior))
        mad = float(np.median(np.abs(prior - m)))
        if mad == 0.0 or np.isnan(mad):
            return np.nan
        z = (float(arr[-1]) - m) / (1.4826 * mad)
        return float(np.clip(z, -clip, clip))

    if isinstance(x, pd.DataFrame):
        return x.apply(lambda col: col.rolling(window=window, min_periods=min_obs).apply(_z, raw=True))
    return x.rolling(window=window, min_periods=min_obs).apply(_z, raw=True)


def cross_sectional_z(x_row: pd.Series) -> pd.Series:
    """Per-day cross-sectional z-score (mean 0, std 1) across the universe."""
    mu = x_row.mean()
    sd = x_row.std(ddof=0)
    if sd == 0 or pd.isna(sd):
        return pd.Series(0.0, index=x_row.index)
    return (x_row - mu) / sd


# ---------- breadth (§3.4) ----------

def pct_advancing(daily_returns: pd.DataFrame) -> pd.Series:
    """% of universe with r_d > 0, per day."""
    return (daily_returns > 0).sum(axis=1) / daily_returns.notna().sum(axis=1)


def ad_diffusion(daily_returns: pd.DataFrame, span: int = 10) -> pd.Series:
    """EMA of (adv − dec) / N."""
    adv = (daily_returns > 0).sum(axis=1)
    dec = (daily_returns < 0).sum(axis=1)
    n = daily_returns.notna().sum(axis=1).replace(0, np.nan)
    raw = (adv - dec) / n
    return raw.ewm(span=span, adjust=False).mean()


def mcclellan_oscillator(daily_returns: pd.DataFrame) -> pd.Series:
    """EMA_19(adv−dec) − EMA_39(adv−dec)."""
    net = (daily_returns > 0).sum(axis=1) - (daily_returns < 0).sum(axis=1)
    return net.ewm(span=19, adjust=False).mean() - net.ewm(span=39, adjust=False).mean()


def concentration(weekly_returns: pd.DataFrame, top_k: int = 5) -> pd.Series:
    """Herfindahl on top-K performers' positive weekly returns.

    > 0.45 with rising indices = narrow leadership warning.
    """
    def _hhi(row: pd.Series) -> float:
        pos = row.clip(lower=0).dropna()
        if pos.sum() == 0:
            return np.nan
        topk = pos.nlargest(top_k)
        w = topk / topk.sum()
        return float((w ** 2).sum())
    return weekly_returns.apply(_hhi, axis=1)


def sector_participation(close: pd.DataFrame, sector_cols: list[str], ma: int = 50) -> pd.Series:
    """Fraction of sector ETFs trading above their `ma`-day moving average."""
    cols = [c for c in sector_cols if c in close.columns]
    if not cols:
        return pd.Series(np.nan, index=close.index)
    sub = close[cols]
    sma = sub.rolling(window=ma, min_periods=int(ma * 0.8)).mean()
    above = (sub > sma)
    return above.sum(axis=1) / sub.notna().sum(axis=1)


# ---------- helpers ----------

def pivot_bars(bars: pd.DataFrame, value: str = "adj_close") -> pd.DataFrame:
    """raw_bars long-form (symbol, ts, ...) -> wide pivot (ts × symbol)."""
    df = bars.pivot(index="ts", columns="symbol", values=value).sort_index()
    df.index = pd.to_datetime(df.index)
    return df
