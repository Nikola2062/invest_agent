"""Forecast Scorecard — wishlist W1.

Every outlook the report publishes is recorded in the `forecasts` table at
publish time and resolved against what actually happened once the horizon
has elapsed. The report's Forecast Scorecard section then shows hit rates
per horizon plus a recent forecast-vs-actual history — the mechanical answer
to "why should I trust this?".

Three forecast types are tracked, matching what the report publishes:

  spy_5d     — Probabilistic Market Forecast, 5-day SPY outlook.
               direction = dominant bucket of the analogue distribution.
  spy_21d    — same at 21 days.
  sector_21d — Sector Forecast (Expected Leaders, Next 21 Days).
               details.predicted_top3 = top-3 buckets by analogue frequency.

Hit rules (also stated in the report so the reader can audit them):
  spy_*      — hit iff the dominant direction equals the realized direction,
               where realized direction uses the SAME cutoff stored with the
               forecast (|log return| > cutoff → bullish/bearish, else neutral).
  sector_21d — hit iff the #1 predicted bucket finishes in the realized top-3
               buckets by mean forward return.

Backfilling is legitimate: `record_forecasts(asof)` only uses analogue
history strictly before `asof` (30-day blackout inside find_analogues), so
recording a forecast for a past date uses exactly the information that was
available then — no lookahead.
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime

import duckdb
import numpy as np
import pandas as pd

from .analogue import forecast_confidence, forecast_distribution, sector_leader_forecast
from .buckets import bucket_for
from .config import Config
from .store import connect
from .validate import latest_verdicts

log = logging.getLogger(__name__)

FORECAST_TYPES = ("spy_5d", "spy_21d", "sector_21d")


def _dominant_direction(f: dict) -> str:
    probs = {
        "bullish": f["bullish_pct"],
        "neutral": f["neutral_pct"],
        "bearish": f["bearish_pct"],
    }
    return max(probs, key=probs.get)


def record_forecasts(cfg: Config, asof: date, verdicts: dict | None = None) -> dict:
    """Persist today's published forecasts. Idempotent per (ts, forecast_type):
    an unresolved row is refreshed; a resolved row is never overwritten (the
    scorecard must not rewrite history as the analogue pool deepens)."""
    if verdicts is None:
        verdicts = latest_verdicts(cfg)

    rows: list[dict] = []
    for ftype, horizon in (("spy_5d", 5), ("spy_21d", 21)):
        f = forecast_distribution(cfg, asof, target="SPY", horizon=horizon, k=30)
        if f.get("bullish_pct") is None:
            continue
        conf = forecast_confidence(f, verdicts=verdicts)
        rows.append({
            "ts": asof, "forecast_type": ftype, "horizon_days": horizon,
            "target": "SPY", "direction": _dominant_direction(f),
            "bullish_pct": f["bullish_pct"], "neutral_pct": f["neutral_pct"],
            "bearish_pct": f["bearish_pct"], "median_fwd": f["median_fwd"],
            "cutoff": f["cutoff"], "confidence": conf["score"],
            "n_analogues": f["n_analogues"], "details": None,
        })

    slf = sector_leader_forecast(cfg, asof, horizon=21, k=30, top_k=5)
    if not slf.empty:
        predicted = slf.head(3).index.tolist()
        rows.append({
            "ts": asof, "forecast_type": "sector_21d", "horizon_days": 21,
            "target": None, "direction": None,
            "bullish_pct": None, "neutral_pct": None, "bearish_pct": None,
            "median_fwd": None, "cutoff": None, "confidence": None,
            "n_analogues": None,
            "details": json.dumps({"predicted_top3": predicted}),
        })

    n_written = 0
    with connect(cfg.storage.duckdb_path) as con:
        for r in rows:
            resolved = con.execute(
                "SELECT resolved_at FROM forecasts WHERE ts = ? AND forecast_type = ?",
                [r["ts"], r["forecast_type"]],
            ).fetchone()
            if resolved is not None and resolved[0] is not None:
                continue  # already resolved — history is immutable
            con.execute(
                "DELETE FROM forecasts WHERE ts = ? AND forecast_type = ?",
                [r["ts"], r["forecast_type"]],
            )
            con.execute(
                """
                INSERT INTO forecasts (ts, forecast_type, horizon_days, target,
                    direction, bullish_pct, neutral_pct, bearish_pct, median_fwd,
                    cutoff, confidence, n_analogues, details, created_at)
                VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                [r["ts"], r["forecast_type"], r["horizon_days"], r["target"],
                 r["direction"], r["bullish_pct"], r["neutral_pct"],
                 r["bearish_pct"], r["median_fwd"], r["cutoff"], r["confidence"],
                 r["n_analogues"], r["details"], datetime.utcnow()],
            )
            n_written += 1
    return {"asof": asof.isoformat(), "recorded": n_written, "candidates": len(rows)}


def _realized_spy(con: duckdb.DuckDBPyConnection, f_ts: date, horizon: int,
                  target: str) -> float | None:
    """Forward `horizon`-trading-day log return of `target` from f_ts, or None
    if the horizon hasn't elapsed in the stored bars yet."""
    bars = con.execute(
        "SELECT ts, adj_close FROM raw_bars WHERE symbol = ? AND ts >= ? "
        "ORDER BY ts LIMIT ?",
        [target, f_ts, horizon + 1],
    ).df()
    if len(bars) < horizon + 1:
        return None
    first_ts = pd.Timestamp(bars["ts"].iloc[0]).date()
    if first_ts != f_ts:
        return None  # no bar on the publish date itself — leave unresolved
    c0 = float(bars["adj_close"].iloc[0])
    ch = float(bars["adj_close"].iloc[horizon])
    if c0 <= 0 or np.isnan(c0) or np.isnan(ch):
        return None
    return float(np.log(ch / c0))


def _realized_top_buckets(con: duckdb.DuckDBPyConnection, f_ts: date,
                          horizon: int, top_k: int = 3) -> list[str] | None:
    """Top-`top_k` buckets by mean per-symbol forward return over the next
    `horizon` trading days after f_ts, or None if not enough days yet."""
    df = con.execute(
        "SELECT ts, symbol, r_d FROM metrics_daily WHERE ts > ? AND r_d IS NOT NULL "
        "ORDER BY symbol, ts",
        [f_ts],
    ).df()
    if df.empty:
        return None
    fwd = (
        df.groupby("symbol")
        .agg(n=("r_d", "size"), fwd=("r_d", lambda s: float(s.head(horizon).sum())))
    )
    fwd = fwd[fwd["n"] >= horizon]
    if fwd.empty:
        return None
    fwd["bucket"] = [bucket_for(s) for s in fwd.index]
    bucket_means = fwd.groupby("bucket")["fwd"].mean().sort_values(ascending=False)
    return bucket_means.head(top_k).index.tolist()


def resolve_forecasts(cfg: Config, asof: date) -> dict:
    """Fill resolution columns for every unresolved forecast whose horizon has
    elapsed in the stored data. Safe to call repeatedly."""
    n_resolved = 0
    with connect(cfg.storage.duckdb_path) as con:
        pending = con.execute(
            "SELECT ts, forecast_type, horizon_days, target, direction, cutoff, details "
            "FROM forecasts WHERE resolved_at IS NULL AND ts < ? ORDER BY ts",
            [asof],
        ).df()
        for _, r in pending.iterrows():
            f_ts = pd.Timestamp(r["ts"]).date()
            horizon = int(r["horizon_days"])
            if r["forecast_type"] == "sector_21d":
                actual_top = _realized_top_buckets(con, f_ts, horizon, top_k=3)
                if actual_top is None:
                    continue
                details = json.loads(r["details"]) if r["details"] else {}
                predicted = details.get("predicted_top3") or []
                hit = bool(predicted and predicted[0] in actual_top)
                details["actual_top3"] = actual_top
                con.execute(
                    "UPDATE forecasts SET resolved_at = ?, actual_value = NULL, "
                    "actual_direction = NULL, hit = ?, details = ? "
                    "WHERE ts = ? AND forecast_type = ?",
                    [datetime.utcnow(), hit, json.dumps(details),
                     r["ts"], r["forecast_type"]],
                )
            else:
                actual = _realized_spy(con, f_ts, horizon, str(r["target"]))
                if actual is None:
                    continue
                cutoff = float(r["cutoff"])
                if actual > cutoff:
                    actual_dir = "bullish"
                elif actual < -cutoff:
                    actual_dir = "bearish"
                else:
                    actual_dir = "neutral"
                hit = bool(str(r["direction"]) == actual_dir)
                con.execute(
                    "UPDATE forecasts SET resolved_at = ?, actual_value = ?, "
                    "actual_direction = ?, hit = ? WHERE ts = ? AND forecast_type = ?",
                    [datetime.utcnow(), actual, actual_dir, hit,
                     r["ts"], r["forecast_type"]],
                )
            n_resolved += 1
    return {"asof": asof.isoformat(), "resolved": n_resolved}


def load_scorecard(con: duckdb.DuckDBPyConnection, asof: date,
                   last_n: int = 100) -> dict:
    """Hit rates per forecast type over the last `last_n` resolved forecasts at
    or before `asof`, plus the most recent resolved rows for the history table."""
    summary: dict[str, dict] = {}
    for ftype in FORECAST_TYPES:
        row = con.execute(
            """
            SELECT COUNT(*) AS n,
                   SUM(CASE WHEN hit THEN 1 ELSE 0 END) AS n_hit
            FROM (
                SELECT hit FROM forecasts
                WHERE forecast_type = ? AND resolved_at IS NOT NULL AND ts <= ?
                ORDER BY ts DESC LIMIT ?
            )
            """,
            [ftype, asof, last_n],
        ).fetchone()
        n, n_hit = int(row[0] or 0), int(row[1] or 0)
        n_pending = con.execute(
            "SELECT COUNT(*) FROM forecasts "
            "WHERE forecast_type = ? AND resolved_at IS NULL AND ts <= ?",
            [ftype, asof],
        ).fetchone()[0]
        summary[ftype] = {
            "n_resolved": n,
            "hit_rate": (n_hit / n) if n else None,
            "n_pending": int(n_pending or 0),
        }

    recent = con.execute(
        """
        SELECT ts, forecast_type, direction, bullish_pct, bearish_pct,
               actual_value, actual_direction, hit, details
        FROM forecasts
        WHERE resolved_at IS NOT NULL AND ts <= ?
        ORDER BY ts DESC, forecast_type LIMIT 9
        """,
        [asof],
    ).df()
    return {"summary": summary, "recent": recent}


def backfill_scorecard(cfg: Config, start: date, end: date) -> dict:
    """Record forecasts for every signal day in [start, end], then resolve.

    No lookahead: each day's forecast only uses analogues strictly before it.
    The verdicts passed to forecast_confidence are today's (validation history
    isn't versioned per day) — confidence is presentation metadata, not part
    of the hit-rate math, so this is acceptable for a backfill.
    """
    verdicts = latest_verdicts(cfg)
    with connect(cfg.storage.duckdb_path) as con:
        days = con.execute(
            "SELECT DISTINCT ts FROM signals_daily WHERE ts >= ? AND ts <= ? ORDER BY ts",
            [start, end],
        ).df()
    n_recorded = 0
    day_list = [pd.Timestamp(t).date() for t in days["ts"]] if not days.empty else []
    for d in day_list:
        out = record_forecasts(cfg, d, verdicts=verdicts)
        n_recorded += out["recorded"]
    res = resolve_forecasts(cfg, end)
    return {
        "start": start.isoformat(), "end": end.isoformat(),
        "days": len(day_list), "recorded": n_recorded,
        "resolved": res["resolved"],
    }
