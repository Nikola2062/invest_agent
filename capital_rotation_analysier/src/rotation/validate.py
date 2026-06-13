"""Signal validation harness per the project docs §3.7.

For each signal we compute:
  1. Rolling rank-IC (Spearman) of today's score vs forward 5d and 21d returns
     of a designated asset/spread. Median IC and pct of windows with IC>0.
  2. Regime-bucketed hit rate: bucket history by SPY 30d realized vol
     (low/mid/high terciles) and check that hit rate ≥ 55% in at least 2/3
     buckets and never < 45% in any bucket.

Gates (a signal passes only if ALL hold):
  - median rolling-IC (5d horizon) >= MIN_MEDIAN_IC
  - pct of rolling windows with IC > 0 >= MIN_PCT_POS_WINDOWS
  - hit rate >= MIN_HIT_BUCKETS in at least N_MIN_GOOD_BUCKETS buckets
  - hit rate >= NEVER_BELOW in every bucket

A signal that fails is tagged 'fail' with a `reason` describing which gate
broke. A signal with too little history is 'undetermined' (not 'fail') so the
operator knows to wait, not investigate.

The §3.7 spec calls for "walk-forward from 2010" — we don't have that history
in MVP. The harness reports its honest sample size and the operator can
re-validate as the backfill grows.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd

from . import metrics as M
from .compute import load_panel
from .config import Config
from .store import connect

log = logging.getLogger(__name__)


# §3.7 gates — exposed here so the operator can tune in one place.
#
# Phase-B (2026-06-09 onward): backfill extended to 2015-01-01 (≥ 2,200 trading
# days for the core universe). The harness now operates in spec mode with the
# 252-day rolling IC window per §3.7. The pragmatic-mode constants are kept
# below as a fallback path for fresh deployments that haven't yet backfilled.
MIN_MEDIAN_IC          = 0.05
MIN_PCT_POS_WINDOWS    = 0.60
ROLLING_IC_WINDOW      = 252
HIT_RATE_GOOD          = 0.55      # need this in N_MIN_GOOD_BUCKETS
HIT_RATE_NEVER_BELOW   = 0.45      # never below this
N_MIN_GOOD_BUCKETS     = 2
MIN_OBS_FOR_VERDICT    = 500       # below this -> undetermined. Need IC window (252) + horizon (21) + buffer.
                                   # Older "pragmatic" value was 120 (paired with the 63d window).

# Which forward-return asset each composite signal is supposed to predict.
# These define the IC test target. Per-asset signals (relative_strength)
# use cross-sectional ranks against each asset's own forward return.
FORWARD_ASSET = {
    "risk_on_off":       "SPY",     # constructive RoO -> SPY up
    "capital_rotation":  "SPY",     # rotation into equities -> SPY up
    "growth":            "SMH",     # growth-leaders proxy
    "recession":         "TLT",     # rising recession -> bonds rally (price up)
    "inflation":         "USO",     # rising inflation -> oil up.
                                    # NOT CPER: copper was in the basket and
                                    # exhibited the strongest mean-reversion of
                                    # any commodity at the monthly horizon —
                                    # see design note S1.INFL (2026-06-10).
    "liquidity":         "SPY",     # easing liquidity -> SPY up
    # 'relative_volume' is a magnitude signal (doesn't predict direction);
    # we validate it differently — see assess_relative_volume.
    # 'relative_strength' is per-asset; cross-sectional treatment.
}


@dataclass
class Verdict:
    signal_name: str
    asof: date
    verdict: str          # 'pass' | 'fail' | 'undetermined'
    reason: str
    median_ic_5d: float | None = None
    median_ic_21d: float | None = None
    pct_windows_pos_ic: float | None = None
    hit_rate_overall: float | None = None
    hit_rate_low_vol: float | None = None
    hit_rate_mid_vol: float | None = None
    hit_rate_high_vol: float | None = None
    n_observations: int = 0
    forward_asset: str | None = None
    details: dict = field(default_factory=dict)


# ============================================================
# Loaders
# ============================================================

def _load_signal_series(con: duckdb.DuckDBPyConnection, name: str) -> pd.Series:
    df = con.execute(
        "SELECT ts, score FROM signals_daily WHERE signal_name = ? ORDER BY ts",
        [name],
    ).df()
    if df.empty:
        return pd.Series(dtype=float)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")["score"].dropna()


def _load_relative_strength_per_asset(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    """Per-asset RS scores live inside signals_daily.components for relative_strength."""
    rows = con.execute(
        "SELECT ts, components FROM signals_daily WHERE signal_name = 'relative_strength' ORDER BY ts"
    ).df()
    out = []
    for _, row in rows.iterrows():
        try:
            c = json.loads(row["components"]) if row["components"] else {}
            pa = c.get("per_asset", {})
            for sym, score in pa.items():
                out.append({"ts": row["ts"], "symbol": sym, "score": score})
        except Exception:
            continue
    if not out:
        return pd.DataFrame()
    df = pd.DataFrame(out)
    df["ts"] = pd.to_datetime(df["ts"])
    return df.pivot(index="ts", columns="symbol", values="score")


def _forward_log_return(close: pd.DataFrame, asset: str, horizon: int) -> pd.Series:
    """ln(C_{t+h} / C_t). Forward-looking; the signal at t should predict this."""
    if asset not in close.columns:
        return pd.Series(dtype=float)
    return np.log(close[asset]).shift(-horizon) - np.log(close[asset])


# ============================================================
# Core IC + hit-rate
# ============================================================

def rolling_spearman_ic(
    score: pd.Series,
    fwd: pd.Series,
    window: int = ROLLING_IC_WINDOW,
) -> pd.Series:
    """Rolling Spearman correlation between score_t and fwd_t over `window` days.

    pandas's .rolling().corr() with method='spearman' is not directly available,
    so we rank within each window and use the Pearson of ranks identity.
    """
    aligned = pd.concat([score, fwd], axis=1, keys=["s", "f"]).dropna()
    if len(aligned) < window:
        return pd.Series(dtype=float)
    s_rank = aligned["s"].rolling(window).rank()
    f_rank = aligned["f"].rolling(window).rank()
    # Spearman == Pearson of ranks
    return s_rank.rolling(window).corr(f_rank)


def hit_rate_buckets(
    score: pd.Series,
    fwd: pd.Series,
    vol_proxy: pd.Series,
) -> dict[str, float]:
    """Bucket history by SPY 30d realized vol (terciles) and report directional hit rate.

    A 'hit' = sign(score) == sign(forward return). For magnitude signals where sign
    is uninformative, this is replaced upstream by a different test (see relative_volume).
    """
    df = pd.concat(
        [score.rename("s"), fwd.rename("f"), vol_proxy.rename("v")],
        axis=1,
    ).dropna()
    if df.empty:
        return {"overall": np.nan, "low": np.nan, "mid": np.nan, "high": np.nan, "n": 0}
    df["hit"] = (np.sign(df["s"]) == np.sign(df["f"])).astype(int)

    # Tercile cuts. Use qcut so each bucket has roughly equal observations.
    try:
        df["bucket"] = pd.qcut(df["v"], q=3, labels=["low", "mid", "high"], duplicates="drop")
    except Exception:
        df["bucket"] = "mid"
    by_bucket = df.groupby("bucket", observed=True)["hit"].mean()
    return {
        "overall": float(df["hit"].mean()),
        "low":     float(by_bucket.get("low", np.nan)) if pd.notna(by_bucket.get("low", np.nan)) else np.nan,
        "mid":     float(by_bucket.get("mid", np.nan)) if pd.notna(by_bucket.get("mid", np.nan)) else np.nan,
        "high":    float(by_bucket.get("high", np.nan)) if pd.notna(by_bucket.get("high", np.nan)) else np.nan,
        "n":       int(len(df)),
    }


# ============================================================
# Per-signal assessment
# ============================================================

def assess_directional(
    name: str,
    asof: date,
    score: pd.Series,
    close: pd.DataFrame,
    asset: str,
    vol_proxy: pd.Series,
) -> Verdict:
    """Standard test for signals that predict direction of a single asset."""
    fwd5 = _forward_log_return(close, asset, 5)
    fwd21 = _forward_log_return(close, asset, 21)

    ic5 = rolling_spearman_ic(score, fwd5)
    ic21 = rolling_spearman_ic(score, fwd21)

    n_obs = int(min(len(score.dropna()), len(fwd5.dropna())))

    if n_obs < MIN_OBS_FOR_VERDICT:
        return Verdict(
            signal_name=name, asof=asof, verdict="undetermined",
            reason=f"insufficient_history (n={n_obs} < {MIN_OBS_FOR_VERDICT})",
            n_observations=n_obs, forward_asset=asset,
        )

    median_ic5 = float(ic5.median()) if not ic5.empty else np.nan
    median_ic21 = float(ic21.median()) if not ic21.empty else np.nan
    pct_pos = float((ic5 > 0).mean()) if not ic5.empty else np.nan

    buckets = hit_rate_buckets(score, fwd5, vol_proxy)
    good_buckets = sum(
        1 for k in ("low", "mid", "high")
        if pd.notna(buckets[k]) and buckets[k] >= HIT_RATE_GOOD
    )
    any_below = any(
        pd.notna(buckets[k]) and buckets[k] < HIT_RATE_NEVER_BELOW
        for k in ("low", "mid", "high")
    )

    reasons = []
    if pd.isna(median_ic5) or median_ic5 < MIN_MEDIAN_IC:
        reasons.append(f"median_ic_5d={median_ic5:.3f} < {MIN_MEDIAN_IC}")
    if pd.isna(pct_pos) or pct_pos < MIN_PCT_POS_WINDOWS:
        reasons.append(f"pct_windows_pos_ic={pct_pos:.2f} < {MIN_PCT_POS_WINDOWS}")
    if good_buckets < N_MIN_GOOD_BUCKETS:
        reasons.append(f"good_vol_buckets={good_buckets} < {N_MIN_GOOD_BUCKETS}")
    if any_below:
        reasons.append(f"hit_rate < {HIT_RATE_NEVER_BELOW} in at least one vol bucket")

    verdict = "pass" if not reasons else "fail"
    reason = "; ".join(reasons) if reasons else "all gates passed"

    return Verdict(
        signal_name=name, asof=asof, verdict=verdict, reason=reason,
        median_ic_5d=None if pd.isna(median_ic5) else median_ic5,
        median_ic_21d=None if pd.isna(median_ic21) else median_ic21,
        pct_windows_pos_ic=None if pd.isna(pct_pos) else pct_pos,
        hit_rate_overall=None if pd.isna(buckets["overall"]) else buckets["overall"],
        hit_rate_low_vol=None if pd.isna(buckets["low"]) else buckets["low"],
        hit_rate_mid_vol=None if pd.isna(buckets["mid"]) else buckets["mid"],
        hit_rate_high_vol=None if pd.isna(buckets["high"]) else buckets["high"],
        n_observations=n_obs, forward_asset=asset,
    )


def assess_relative_strength(
    asof: date,
    rs_panel: pd.DataFrame,    # per-asset RS scores indexed by date
    close: pd.DataFrame,
    vol_proxy: pd.Series,
) -> Verdict:
    """Cross-sectional rank-IC: rank assets by today's RS, rank by forward return.

    A useful per-asset signal should have positive cross-sectional rank
    correlation: high-RS names should produce higher forward returns on average.
    """
    if rs_panel.empty:
        return Verdict("relative_strength", asof, "undetermined",
                       "no_per_asset_rs_history", n_observations=0)

    common = sorted(set(rs_panel.columns) & set(close.columns))
    if len(common) < 5:
        return Verdict("relative_strength", asof, "undetermined",
                       f"too_few_assets_overlapping (n={len(common)})", n_observations=0)

    fwd = pd.DataFrame({s: _forward_log_return(close, s, 5) for s in common})
    rs = rs_panel[common]
    # Cross-sectional Spearman per day (rank both axes within row, take Pearson)
    daily_ic = []
    for ts in rs.index.intersection(fwd.index):
        r = rs.loc[ts].dropna()
        f = fwd.loc[ts].dropna()
        common_syms = r.index.intersection(f.index)
        if len(common_syms) < 5:
            continue
        rr = r.loc[common_syms].rank()
        ff = f.loc[common_syms].rank()
        if rr.std() == 0 or ff.std() == 0:
            continue
        ic = float(rr.corr(ff))
        if not np.isnan(ic):
            daily_ic.append(ic)

    n_obs = len(daily_ic)
    if n_obs < MIN_OBS_FOR_VERDICT:
        return Verdict("relative_strength", asof, "undetermined",
                       f"insufficient_history (n={n_obs} < {MIN_OBS_FOR_VERDICT})",
                       n_observations=n_obs)

    median_ic = float(np.median(daily_ic))
    pct_pos = float(np.mean([x > 0 for x in daily_ic]))

    reasons = []
    if median_ic < MIN_MEDIAN_IC:
        reasons.append(f"median_cs_ic={median_ic:.3f} < {MIN_MEDIAN_IC}")
    if pct_pos < MIN_PCT_POS_WINDOWS:
        reasons.append(f"pct_days_pos_ic={pct_pos:.2f} < {MIN_PCT_POS_WINDOWS}")

    return Verdict(
        signal_name="relative_strength", asof=asof,
        verdict="pass" if not reasons else "fail",
        reason="; ".join(reasons) if reasons else "all gates passed",
        median_ic_5d=median_ic, pct_windows_pos_ic=pct_pos,
        n_observations=n_obs, forward_asset="cross_sectional",
        details={"approach": "cross_sectional_daily_spearman"},
    )


def assess_relative_volume(
    asof: date,
    score: pd.Series,
    close: pd.DataFrame,
) -> Verdict:
    """Relative Volume is a magnitude/event signal, not directional. Validate by:
    high RV should precede above-median forward realized vol (i.e. RV detects
    'something is happening' even if direction is uncertain).
    """
    spy = close.get("SPY")
    if spy is None or score.empty:
        return Verdict("relative_volume", asof, "undetermined",
                       "no_spy_or_score_history", n_observations=0)

    fwd_vol = M.realized_vol(close[["SPY"]], 5)["SPY"].shift(-5)
    aligned = pd.concat([score.rename("s"), fwd_vol.rename("v")], axis=1).dropna()
    n_obs = len(aligned)
    if n_obs < MIN_OBS_FOR_VERDICT:
        return Verdict("relative_volume", asof, "undetermined",
                       f"insufficient_history (n={n_obs} < {MIN_OBS_FOR_VERDICT})",
                       n_observations=n_obs)

    # High RV (top tercile) should produce above-median forward 5d realized vol.
    aligned["s_bucket"] = pd.qcut(aligned["s"], 3, labels=["low", "mid", "high"], duplicates="drop")
    med = aligned["v"].median()
    aligned["above_med_vol"] = (aligned["v"] > med).astype(int)
    rates = aligned.groupby("s_bucket", observed=True)["above_med_vol"].mean().to_dict()

    high_rate = rates.get("high", np.nan)
    low_rate = rates.get("low", np.nan)

    # Pass if high-RV bucket exceeds low-RV bucket by ≥ 10 percentage points.
    reasons = []
    if pd.isna(high_rate) or pd.isna(low_rate):
        reasons.append("buckets_missing")
    elif (high_rate - low_rate) < 0.10:
        reasons.append(f"high_minus_low_above_med_vol={high_rate - low_rate:.2f} < 0.10")

    return Verdict(
        signal_name="relative_volume", asof=asof,
        verdict="pass" if not reasons else "fail",
        reason="; ".join(reasons) if reasons else "high-RV predicts above-median forward vol",
        hit_rate_overall=float(high_rate) if pd.notna(high_rate) else None,
        n_observations=n_obs, forward_asset="SPY_realized_vol",
        details={"approach": "magnitude_predicts_forward_vol",
                 "rates_by_bucket": {k: float(v) for k, v in rates.items() if pd.notna(v)}},
    )


# ============================================================
# Orchestration
# ============================================================

def run_validation(cfg: Config, asof: date | None = None) -> list[Verdict]:
    """Validate every signal against current available history."""
    asof = asof or date.today()
    verdicts: list[Verdict] = []

    with connect(cfg.storage.duckdb_path) as con:
        close, _ = load_panel(con, asof, lookback_days=3000)
        if close.empty:
            log.warning("validate: no bar history")
            return []

        spy_vol = M.realized_vol(close[["SPY"]], 30)["SPY"] if "SPY" in close.columns else pd.Series(dtype=float)

        for name, asset in FORWARD_ASSET.items():
            score = _load_signal_series(con, name)
            v = assess_directional(name, asof, score, close, asset, spy_vol)
            verdicts.append(v)

        # Relative volume (magnitude)
        rv_score = _load_signal_series(con, "relative_volume")
        verdicts.append(assess_relative_volume(asof, rv_score, close))

        # Relative strength (cross-sectional, per-asset)
        rs_panel = _load_relative_strength_per_asset(con)
        verdicts.append(assess_relative_strength(asof, rs_panel, close, spy_vol))

        # Persist
        now = datetime.utcnow()
        for v in verdicts:
            con.execute(
                "DELETE FROM signal_validation WHERE signal_name = ? AND asof_date = ?",
                [v.signal_name, v.asof],
            )
            con.execute(
                """
                INSERT INTO signal_validation
                (signal_name, asof_date, verdict, reason,
                 median_ic_5d, median_ic_21d, pct_windows_pos_ic,
                 hit_rate_overall, hit_rate_low_vol, hit_rate_mid_vol, hit_rate_high_vol,
                 n_observations, forward_asset, details, computed_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [v.signal_name, v.asof, v.verdict, v.reason,
                 v.median_ic_5d, v.median_ic_21d, v.pct_windows_pos_ic,
                 v.hit_rate_overall, v.hit_rate_low_vol, v.hit_rate_mid_vol, v.hit_rate_high_vol,
                 v.n_observations, v.forward_asset,
                 json.dumps(v.details), now],
            )

    return verdicts


def latest_verdicts(cfg: Config) -> dict[str, Verdict]:
    """Returns the most recent verdict per signal, used by report+alerts at runtime."""
    out: dict[str, Verdict] = {}
    with connect(cfg.storage.duckdb_path) as con:
        df = con.execute(
            """
            SELECT signal_name, asof_date, verdict, reason,
                   median_ic_5d, median_ic_21d, pct_windows_pos_ic,
                   hit_rate_overall, hit_rate_low_vol, hit_rate_mid_vol, hit_rate_high_vol,
                   n_observations, forward_asset, details
            FROM signal_validation sv
            WHERE asof_date = (SELECT MAX(asof_date) FROM signal_validation WHERE signal_name = sv.signal_name)
            """
        ).df()
    for _, r in df.iterrows():
        out[r["signal_name"]] = Verdict(
            signal_name=r["signal_name"],
            asof=r["asof_date"] if isinstance(r["asof_date"], date) else r["asof_date"].date(),
            verdict=r["verdict"], reason=r["reason"] or "",
            median_ic_5d=None if pd.isna(r["median_ic_5d"]) else float(r["median_ic_5d"]),
            median_ic_21d=None if pd.isna(r["median_ic_21d"]) else float(r["median_ic_21d"]),
            pct_windows_pos_ic=None if pd.isna(r["pct_windows_pos_ic"]) else float(r["pct_windows_pos_ic"]),
            hit_rate_overall=None if pd.isna(r["hit_rate_overall"]) else float(r["hit_rate_overall"]),
            hit_rate_low_vol=None if pd.isna(r["hit_rate_low_vol"]) else float(r["hit_rate_low_vol"]),
            hit_rate_mid_vol=None if pd.isna(r["hit_rate_mid_vol"]) else float(r["hit_rate_mid_vol"]),
            hit_rate_high_vol=None if pd.isna(r["hit_rate_high_vol"]) else float(r["hit_rate_high_vol"]),
            n_observations=int(r["n_observations"]) if not pd.isna(r["n_observations"]) else 0,
            forward_asset=r["forward_asset"],
            details=json.loads(r["details"]) if r["details"] else {},
        )
    return out
