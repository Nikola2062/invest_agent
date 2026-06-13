"""Historical Analogue Engine — Phase C, wishlist item 10.

Given today's 8-signal vector, find the K most-similar historical days by
cosine similarity. The "analogue forward return" is then the distribution of
what those similar days actually produced over the next N trading days. This
turns "what happened next in similar regimes?" into a tractable lookup.

Inputs (all in DuckDB):
  - signals_daily : the 8 composite scores per date — defines the feature vector
  - raw_bars      : per-asset adjusted closes — provides forward returns
  - regime_history: per-date regime label — provides the regime context for each analogue

Outputs (returned as plain dicts so the report layer can render them):
  - find_analogues(asof, k=5, blackout=30) -> list[Analogue]
  - regime_transition_matrix(window=30) -> DataFrame (next-N-day regime probs)
  - rotation_probability_matrix(asof, k=20, horizon=5) -> DataFrame (bucket probs)

The "blackout" window prevents trivially-similar adjacent days from being
selected as analogues (e.g. asking for the most similar day to 2026-06-05
shouldn't return 2026-06-04). Default 30 trading days = ~6 calendar weeks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

from .buckets import bucket_for, group_metrics_by_bucket
from .config import Config
from .store import connect


# Composite signals used as the feature vector. Order is fixed so the cached
# numpy array is stable across runs. The per-asset relative_strength is
# omitted because its headline is the top-|z| asset (changes meaning by row).
ANALOGUE_SIGNALS: tuple[str, ...] = (
    "capital_rotation",
    "risk_on_off",
    "inflation",
    "growth",
    "recession",
    "liquidity",
    "relative_volume",
)


@dataclass
class Analogue:
    asof: date
    similarity: float                # 1.0 = identical, 0.0 = orthogonal
    regime: str | None = None
    days_in_regime: int | None = None
    fwd_return_spy_5d: float | None = None
    fwd_return_spy_21d: float | None = None
    next_regime_30d: str | None = None
    bucket_winners_5d: list[str] = field(default_factory=list)


# ============================================================
# Internal loaders
# ============================================================


def _load_signal_matrix(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Wide DataFrame: rows = dates, cols = ANALOGUE_SIGNALS.

    NaN rows are dropped — we only compare days that have ALL signals present.
    """
    df = con.execute(
        "SELECT ts, signal_name, score FROM signals_daily "
        "WHERE signal_name IN ({}) AND score IS NOT NULL".format(
            ",".join(f"'{s}'" for s in ANALOGUE_SIGNALS)
        )
    ).df()
    if df.empty:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"])
    wide = df.pivot(index="ts", columns="signal_name", values="score")
    # Maintain a stable column order; missing signals stay NaN.
    return wide.reindex(columns=list(ANALOGUE_SIGNALS)).sort_index()


def _load_regime_panel(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    df = con.execute(
        "SELECT ts, regime, days_in_regime FROM regime_history ORDER BY ts"
    ).df()
    if df.empty:
        return pd.DataFrame(columns=["regime", "days_in_regime"])
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")[["regime", "days_in_regime"]]


def _load_forward_returns(
    con: duckdb.DuckDBPyConnection,
    symbol: str,
    horizons: Iterable[int] = (5, 21),
) -> pd.DataFrame:
    df = con.execute(
        "SELECT ts, adj_close FROM raw_bars WHERE symbol = ? ORDER BY ts",
        [symbol],
    ).df()
    if df.empty:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts")
    out = pd.DataFrame(index=df.index)
    for h in horizons:
        # Forward log return: ln(C_{t+h} / C_t). Shift by -h to align with t.
        out[f"fwd_{h}d"] = np.log(df["adj_close"].shift(-h) / df["adj_close"])
    return out


# ============================================================
# Similarity primitive
# ============================================================


def _cosine_similarity_against(target: np.ndarray, panel: np.ndarray) -> np.ndarray:
    """Cosine similarity between `target` (D,) and every row in `panel` (N×D).

    Both inputs may have NaN entries; rows containing any NaN return -inf
    similarity so they're never selected as analogues.
    """
    valid_mask = ~np.isnan(panel).any(axis=1) & ~np.isnan(target).any()
    out = np.full(panel.shape[0], -np.inf)
    if not valid_mask.any():
        return out
    target_norm = float(np.linalg.norm(target))
    if target_norm == 0:
        return out
    panel_norm = np.linalg.norm(panel[valid_mask], axis=1)
    safe = panel_norm > 0
    dots = panel[valid_mask] @ target
    sims = np.full(valid_mask.sum(), -np.inf)
    sims[safe] = dots[safe] / (panel_norm[safe] * target_norm)
    out[valid_mask] = sims
    return out


# ============================================================
# Public API
# ============================================================


def find_analogues(
    cfg: Config,
    asof: date,
    k: int = 5,
    blackout: int = 30,
    forward_symbol: str = "SPY",
) -> list[Analogue]:
    """Return the k most-similar historical days to `asof`.

    Args:
      asof: target date.
      k: number of analogues to return.
      blackout: skip the trailing N trading days around `asof` so adjacent
                days aren't selected as their own analogues.
      forward_symbol: which asset's forward returns to attach to the analogue.

    Returns a list ordered by descending similarity.
    """
    target_ts = pd.Timestamp(asof)

    with connect(cfg.storage.duckdb_path) as con:
        sig_mat = _load_signal_matrix(con)
        if sig_mat.empty or target_ts not in sig_mat.index:
            return []

        target_vec = sig_mat.loc[target_ts].to_numpy(dtype=float)
        # Candidate pool: everything before the blackout window
        cutoff = target_ts - pd.Timedelta(days=blackout)
        candidates = sig_mat[sig_mat.index < cutoff]
        if candidates.empty:
            return []

        sims = _cosine_similarity_against(
            target_vec, candidates.to_numpy(dtype=float)
        )
        order = np.argsort(-sims)[:k]
        top_dates = candidates.index[order]
        top_sims = sims[order]

        regime_panel = _load_regime_panel(con)
        fwd_returns = _load_forward_returns(con, forward_symbol, horizons=(5, 21))
        metrics_5d = _load_bucket_forward_winners(con, horizon=5)

        results: list[Analogue] = []
        for ts, sim in zip(top_dates, top_sims):
            ts_date = ts.date() if hasattr(ts, "date") else ts
            reg, dur = (None, None)
            if not regime_panel.empty and ts in regime_panel.index:
                row = regime_panel.loc[ts]
                reg = str(row["regime"]) if pd.notna(row["regime"]) else None
                dur = int(row["days_in_regime"]) if pd.notna(row["days_in_regime"]) else None
            fwd5 = float(fwd_returns.loc[ts, "fwd_5d"]) if ts in fwd_returns.index else None
            fwd21 = float(fwd_returns.loc[ts, "fwd_21d"]) if ts in fwd_returns.index else None
            fwd5  = None if (fwd5 is not None and np.isnan(fwd5))  else fwd5
            fwd21 = None if (fwd21 is not None and np.isnan(fwd21)) else fwd21
            # Where did the regime go 30 trading days later?
            next_30 = None
            if not regime_panel.empty:
                later = regime_panel[regime_panel.index >= ts + pd.Timedelta(days=42)]
                if not later.empty:
                    next_30 = str(later.iloc[0]["regime"])
            winners = metrics_5d.get(ts, [])

            results.append(Analogue(
                asof=ts_date,
                similarity=float(sim) if np.isfinite(sim) else 0.0,
                regime=reg,
                days_in_regime=dur,
                fwd_return_spy_5d=fwd5,
                fwd_return_spy_21d=fwd21,
                next_regime_30d=next_30,
                bucket_winners_5d=winners,
            ))
        return results


def _load_bucket_forward_winners(
    con: duckdb.DuckDBPyConnection,
    horizon: int = 5,
    top_k: int = 3,
) -> dict[pd.Timestamp, list[str]]:
    """For each date in metrics_daily, identify the top-K buckets by mean
    forward `horizon`-day log return across constituents. Used to attach a
    "what led after" tag to each analogue.
    """
    df = con.execute(
        "SELECT ts, symbol, r_d FROM metrics_daily WHERE r_d IS NOT NULL ORDER BY ts, symbol"
    ).df()
    if df.empty:
        return {}
    df["ts"] = pd.to_datetime(df["ts"])
    # Per-symbol forward log return = sum of next `horizon` daily log returns
    df = df.sort_values(["symbol", "ts"])
    df["fwd"] = df.groupby("symbol")["r_d"].transform(
        lambda s: s.shift(-1).rolling(horizon, min_periods=horizon).sum()
    )
    df["bucket"] = df["symbol"].map(bucket_for)
    bucket_fwd = df.groupby(["ts", "bucket"])["fwd"].mean().reset_index()
    bucket_fwd = bucket_fwd.dropna(subset=["fwd"])

    out: dict[pd.Timestamp, list[str]] = {}
    for ts_val, group in bucket_fwd.groupby("ts"):
        top = group.sort_values("fwd", ascending=False).head(top_k)["bucket"].tolist()
        out[ts_val] = top
    return out


def regime_transition_matrix(
    cfg: Config,
    window: int = 30,
) -> pd.DataFrame:
    """P(regime_{t+window} | regime_t), estimated from regime_history.

    Returns a DataFrame indexed by source regime, columns are destination
    regimes; values are probabilities summing to 1.0 along each row.
    """
    with connect(cfg.storage.duckdb_path) as con:
        df = con.execute(
            "SELECT ts, regime FROM regime_history ORDER BY ts"
        ).df()
    if df.empty:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.set_index("ts").sort_index()
    src = df["regime"]
    dst = df["regime"].shift(-window)
    paired = pd.DataFrame({"src": src, "dst": dst}).dropna()
    if paired.empty:
        return pd.DataFrame()
    counts = paired.groupby(["src", "dst"]).size().unstack(fill_value=0)
    return counts.div(counts.sum(axis=1), axis=0)


# ============================================================
# Phase D — Probabilistic forecasts
# ============================================================


# Default cutoffs for the Bullish/Neutral/Bearish bucketing of forward returns.
# Half-percent threshold per 5-day horizon means a bucket needs > 50 bps to be
# called directional, which lines up with the bucket-flat threshold in Phase A.
DEFAULT_DIRECTION_CUTOFF_5D  = 0.005    # 50 bps log return
DEFAULT_DIRECTION_CUTOFF_21D = 0.015    # 150 bps log return


def forecast_distribution(
    cfg: Config,
    asof: date,
    target: str = "SPY",
    horizon: int = 5,
    k: int = 30,
    direction_cutoff: float | None = None,
) -> dict:
    """Probabilistic forecast for `target` over the next `horizon` trading days.

    Mechanism: pull the k most-similar historical analogues; look at what
    actually happened to `target` over the next `horizon` days after each
    analogue; bucket those forward returns into bullish / neutral / bearish.
    The output is the empirical distribution, not a prediction.

    Args:
      target: ticker whose forward return we're bucketing.
      horizon: 5 or 21 trading days are the canonical choices.
      k: how many analogues to draw on; 20-30 balances signal and noise.
      direction_cutoff: log-return magnitude above which a window is called
        bullish/bearish (otherwise neutral). Defaults to a horizon-scaled
        threshold so the buckets stay balanced.

    Returns dict keys:
      bullish_pct, neutral_pct, bearish_pct  — sum to 1.0
      mean_fwd, median_fwd, p10_fwd, p90_fwd — log returns
      n_analogues                            — actual k used (after filtering)
      top_similarity                         — best analogue's similarity
      agreement                              — fraction in the dominant bucket
      target, horizon, cutoff                — passthrough metadata
    """
    if direction_cutoff is None:
        direction_cutoff = (DEFAULT_DIRECTION_CUTOFF_21D
                            if horizon >= 15
                            else DEFAULT_DIRECTION_CUTOFF_5D)

    anas = find_analogues(cfg, asof, k=k, blackout=30, forward_symbol=target)
    if not anas:
        return _empty_forecast(target, horizon, direction_cutoff)

    # Each analogue carries fwd5d / fwd21d for the SPY default. For arbitrary
    # `target` we re-pull the right column on demand.
    if target == "SPY" and horizon in (5, 21):
        key = "fwd_return_spy_5d" if horizon == 5 else "fwd_return_spy_21d"
        fwds = [getattr(a, key) for a in anas]
    else:
        with connect(cfg.storage.duckdb_path) as con:
            fr = _load_forward_returns(con, target, horizons=(horizon,))
        col = f"fwd_{horizon}d"
        if col not in fr.columns:
            return _empty_forecast(target, horizon, direction_cutoff)
        fwds = [
            (float(fr.loc[pd.Timestamp(a.asof), col])
             if pd.Timestamp(a.asof) in fr.index and pd.notna(fr.loc[pd.Timestamp(a.asof), col])
             else None)
            for a in anas
        ]

    valid = [x for x in fwds if x is not None and not np.isnan(x)]
    if not valid:
        return _empty_forecast(target, horizon, direction_cutoff)

    arr = np.array(valid, dtype=float)
    bullish = float((arr >  direction_cutoff).mean())
    bearish = float((arr < -direction_cutoff).mean())
    neutral = 1.0 - bullish - bearish

    dominant = max(bullish, neutral, bearish)

    return {
        "target":         target,
        "horizon":        horizon,
        "cutoff":         direction_cutoff,
        "bullish_pct":    bullish,
        "neutral_pct":    neutral,
        "bearish_pct":    bearish,
        "mean_fwd":       float(arr.mean()),
        "median_fwd":     float(np.median(arr)),
        "p10_fwd":        float(np.percentile(arr, 10)),
        "p90_fwd":        float(np.percentile(arr, 90)),
        "n_analogues":    len(valid),
        "top_similarity": float(anas[0].similarity) if anas else 0.0,
        "agreement":      float(dominant),
    }


def _empty_forecast(target: str, horizon: int, cutoff: float) -> dict:
    return {
        "target": target, "horizon": horizon, "cutoff": cutoff,
        "bullish_pct": None, "neutral_pct": None, "bearish_pct": None,
        "mean_fwd": None, "median_fwd": None, "p10_fwd": None, "p90_fwd": None,
        "n_analogues": 0, "top_similarity": 0.0, "agreement": None,
    }


def sector_leader_forecast(
    cfg: Config,
    asof: date,
    horizon: int = 21,
    k: int = 30,
    top_k: int = 5,
) -> pd.DataFrame:
    """For today's k analogues, fraction of windows where each bucket was a
    top-`top_k` performer over the next `horizon` days.

    Differs from `rotation_probability_matrix` only in defaults: longer
    horizon (21 instead of 5) and broader top-k (5 instead of 3). The
    wishlist's item-13 framing is "Expected Leaders (Next 2-4 Weeks)" which
    matches 21d / top-5.
    """
    # Reuse the same machinery; just swap defaults.
    return rotation_probability_matrix(cfg, asof, k=k, horizon=horizon).head(top_k * 2)


def forecast_confidence(
    forecast: dict,
    verdicts: dict | None = None,
    underlying_signals: tuple[str, ...] = ("capital_rotation", "risk_on_off", "growth"),
) -> dict:
    """Synthesize a 0-1 confidence score for a forecast dict.

    Three inputs (each scaled to [0, 1]):
      1. top_similarity      — best analogue's cosine sim (already 0..1 for
                                positive similarities; negative is treated as 0)
      2. agreement           — fraction of analogues in the dominant direction
      3. validation_credit   — graded per-signal credit. A signal that PASSES
                                IC validation gets 1.0; one that fails strict
                                IC but still has hit_rate ≥ 0.55 gets 0.5
                                (directional but failing at this horizon —
                                still informative for the analogue layer);
                                outright failure gets 0.0. If `verdicts` is
                                None or empty, contributes neutrally (0.5).

    Equal-weighted average, then bucketed:
      ≥0.70 high, 0.50-0.69 medium, 0.35-0.49 low, <0.35 below floor.

    The bucketing matches the §4.5 confidence thresholds used elsewhere so
    operators have a single mental model. The graded scale exists because the
    strict 5d-IC gate is intentionally hostile — a signal can have a 60% hit
    rate (real directional signal) but still fail the gate, and that's
    information worth keeping in the confidence math.
    """
    sim = max(0.0, float(forecast.get("top_similarity") or 0.0))
    agr = float(forecast.get("agreement") or 0.0)

    if verdicts:
        credit_sum = 0.0
        for name in underlying_signals:
            v = verdicts.get(name)
            if v is None:
                credit_sum += 0.5     # untested -> neutral
            elif v.verdict == "pass":
                credit_sum += 1.0
            elif v.hit_rate_overall is not None and v.hit_rate_overall >= 0.55:
                credit_sum += 0.5     # directional, fails strict IC
            else:
                credit_sum += 0.0
        validation_credit = credit_sum / len(underlying_signals)
    else:
        validation_credit = 0.5

    score = (sim + agr + validation_credit) / 3.0

    if score >= 0.70:
        bucket = "high"
    elif score >= 0.50:
        bucket = "medium"
    elif score >= 0.35:
        bucket = "low"
    else:
        bucket = "below_floor"

    return {
        "score": float(score),
        "bucket": bucket,
        "components": {
            "top_similarity":    sim,
            "agreement":         agr,
            "validation_credit": validation_credit,
        },
    }


def rotation_probability_matrix(
    cfg: Config,
    asof: date,
    k: int = 20,
    horizon: int = 5,
) -> pd.DataFrame:
    """For today's k analogues, count which buckets led most often over the
    next `horizon` days. Returns a DataFrame indexed by bucket with columns
    `frequency` (0..1) and `avg_fwd_return` (mean log return of that bucket
    across the k analogue forward windows).

    This is the wishlist's Rotation Probability Matrix — "given today's state,
    what historically tended to be the destination of capital next?"
    """
    analogues = find_analogues(cfg, asof, k=k)
    if not analogues:
        return pd.DataFrame(columns=["frequency", "avg_fwd_return"])

    with connect(cfg.storage.duckdb_path) as con:
        # Aggregate forward-bucket returns from the analogue dates.
        ana_ts = [pd.Timestamp(a.asof) for a in analogues]
        if not ana_ts:
            return pd.DataFrame()
        placeholders = ",".join(["?"] * len(ana_ts))
        rows = con.execute(
            f"SELECT ts, symbol, r_d FROM metrics_daily WHERE ts IN ({placeholders}) "
            "OR ts > (SELECT MIN(ts) FROM metrics_daily WHERE ts IN ({}))".format(placeholders),
            ana_ts + ana_ts,
        ).df()
    if rows.empty:
        return pd.DataFrame()

    # Compute per-symbol per-asof forward `horizon`-day log return.
    rows["ts"] = pd.to_datetime(rows["ts"])
    rows = rows.sort_values(["symbol", "ts"])
    rows["fwd"] = rows.groupby("symbol")["r_d"].transform(
        lambda s: s.shift(-1).rolling(horizon, min_periods=horizon).sum()
    )
    rows["bucket"] = rows["symbol"].map(bucket_for)
    rows = rows[rows["ts"].isin(ana_ts)].dropna(subset=["fwd"])
    if rows.empty:
        return pd.DataFrame()

    # Frequency = fraction of analogues where this bucket was in the top-3.
    # avg_fwd_return = mean forward return across analogues.
    top_k_per_asof: list[list[str]] = []
    for ts_val, group in rows.groupby("ts"):
        bucket_means = group.groupby("bucket")["fwd"].mean()
        top = bucket_means.sort_values(ascending=False).head(3).index.tolist()
        top_k_per_asof.append(top)
    n_asofs = len(top_k_per_asof)

    bucket_counts: dict[str, int] = {}
    for top in top_k_per_asof:
        for b in top:
            bucket_counts[b] = bucket_counts.get(b, 0) + 1
    freq = {b: c / n_asofs for b, c in bucket_counts.items()}

    avg_fwd = rows.groupby("bucket")["fwd"].mean().to_dict()
    all_buckets = sorted(set(freq) | set(avg_fwd))
    out = pd.DataFrame({
        "frequency":      [freq.get(b, 0.0) for b in all_buckets],
        "avg_fwd_return": [avg_fwd.get(b, np.nan) for b in all_buckets],
    }, index=pd.Index(all_buckets, name="bucket"))
    return out.sort_values("frequency", ascending=False)
