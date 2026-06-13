"""Regime classifier per the project docs §4.

Deterministic rule tree maps the 8 signal scores into 5 regimes with hysteresis
to prevent whipsaw. Stores results in regime_history.

Hysteresis: a flip requires N consecutive days of the new regime signature.
N defaults: daily=3, weekly=2, monthly=1.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date, datetime
from typing import Iterable

import duckdb
import pandas as pd

from .config import Config
from .store import connect


REGIME_RISK_ON      = "Risk-On Expansion"
REGIME_LATE_CYCLE   = "Late-Cycle Inflation"
REGIME_RISK_OFF     = "Risk-Off Defensive"
REGIME_DEFLATION    = "Deflationary Shock"
REGIME_UNCERTAIN    = "Regime Uncertain"

HYSTERESIS_DAYS = {"daily": 2, "weekly": 1, "monthly": 1}
SMOOTHING_WINDOW = 5  # trailing days; regimes are multi-day phenomena (§4.3)


def classify(scores: dict[str, float]) -> tuple[str, float, dict]:
    """Returns (regime_name, confidence, components_dict).

    Decision tree precedence (first matching rule wins):
      1. Strong risk-off + high recession → REGIME_RISK_OFF
      2. Risk-off + falling inflation     → REGIME_DEFLATION
      3. Risk-on + rising inflation       → REGIME_LATE_CYCLE
      4. Risk-on + growth >0              → REGIME_RISK_ON
      5. Otherwise                        → REGIME_UNCERTAIN

    Inputs are RAW (or smoothed) scores. Confidence-weighting at this layer
    causes mid-confidence days to drop below threshold and flicker the regime;
    we apply smoothing upstream (5d median) and use raw scores here. The regime's
    own confidence is derived from score magnitudes below.
    """
    roo = scores.get("risk_on_off", 0.0) or 0.0
    rec = scores.get("recession", 0.0) or 0.0
    inf = scores.get("inflation", 0.0) or 0.0
    grw = scores.get("growth", 0.0) or 0.0
    cr  = scores.get("capital_rotation", 0.0) or 0.0
    liq = scores.get("liquidity", 0.0) or 0.0

    comp = {"risk_on_off": roo, "recession": rec, "inflation": inf,
            "growth": grw, "capital_rotation": cr, "liquidity": liq}

    if roo < -25 and rec > 15:
        return REGIME_RISK_OFF, min(1.0, (abs(roo) + rec) / 100.0), comp
    if roo < -20 and inf < -15:
        return REGIME_DEFLATION, min(1.0, (abs(roo) + abs(inf)) / 100.0), comp
    if roo > 20 and inf > 20:
        return REGIME_LATE_CYCLE, min(1.0, (roo + inf) / 100.0), comp
    if roo > 15 and grw > 10:
        return REGIME_RISK_ON, min(1.0, (roo + grw) / 100.0), comp
    return REGIME_UNCERTAIN, 0.30, comp


def _smoothed_scores(con: duckdb.DuckDBPyConnection, asof: date) -> dict[str, float]:
    """Trailing-N-day median of each signal's score ending on asof (inclusive)."""
    # 3x buffer for non-trading days; compute boundary in Python to dodge
    # DuckDB's awkward parameterized INTERVAL syntax.
    start = (pd.Timestamp(asof) - pd.Timedelta(days=SMOOTHING_WINDOW * 3)).date()
    df = con.execute(
        """
        SELECT signal_name, MEDIAN(score) AS med
        FROM signals_daily
        WHERE ts <= ? AND ts > ?
        GROUP BY signal_name
        """,
        [asof, start],
    ).df()
    return {r["signal_name"]: None if pd.isna(r["med"]) else float(r["med"]) for _, r in df.iterrows()}


def _load_last_regime(con: duckdb.DuckDBPyConnection, before: date) -> tuple[str | None, int]:
    df = con.execute(
        "SELECT regime, days_in_regime FROM regime_history "
        "WHERE ts < ? ORDER BY ts DESC LIMIT 1",
        [before],
    ).df()
    if df.empty:
        return None, 0
    row = df.iloc[0]
    return row["regime"], int(row["days_in_regime"]) if row["days_in_regime"] is not None else 0


def _load_recent_proposed(
    con: duckdb.DuckDBPyConnection,
    asof: date,
    lookback_days: int,
) -> list[str]:
    """Return the proposed regimes (un-hysteresised) from the last N classifier runs.

    For hysteresis we look at the candidate regime each day; if the same new regime
    has held for `hysteresis_days` days, we accept the flip.
    """
    # We classify on signals_daily, not regime_history, so we re-derive proposals.
    df = con.execute(
        "SELECT ts FROM (SELECT DISTINCT ts FROM signals_daily WHERE ts <= ? ORDER BY ts DESC LIMIT ?) ORDER BY ts",
        [asof, lookback_days],
    ).df()
    return list(df["ts"])


def _signals_at(con: duckdb.DuckDBPyConnection, ts) -> dict[str, dict]:
    rows = con.execute(
        "SELECT signal_name, score, confidence FROM signals_daily WHERE ts = ?",
        [ts],
    ).df()
    return {
        r["signal_name"]: {
            "score": None if pd.isna(r["score"]) else float(r["score"]),
            "confidence": None if pd.isna(r["confidence"]) else float(r["confidence"]),
        }
        for _, r in rows.iterrows()
    }


def _propose_at(con: duckdb.DuckDBPyConnection, ts: date) -> tuple[str, float, dict] | None:
    if not _signals_at(con, ts):
        return None
    smoothed = _smoothed_scores(con, ts)
    if not smoothed:
        return None
    return classify(smoothed)


def backfill_regimes(cfg: Config, start: date, end: date, horizon: str = "daily") -> dict:
    """Recompute the regime classifier for every day in [start, end].

    Per-day `run_regime_for_date` does ~6 small SQL round-trips. For a 10-year
    backfill that's 60k queries. This function loads the entire signals_daily
    range once, computes the 5d-median in pandas, applies `classify`, and
    bulk-writes the regime_history table. Hysteresis is honored by walking
    the dates in order and tracking committed regime + days_in_regime in
    Python state.
    """
    hyst = HYSTERESIS_DAYS.get(horizon, 2)
    ts_start = pd.Timestamp(start)
    ts_end = pd.Timestamp(end)

    with connect(cfg.storage.duckdb_path) as con:
        # Pull signals over [start - SMOOTHING_WINDOW*3, end] so the leading
        # rolling-median has a non-empty input window.
        load_start = (ts_start - pd.Timedelta(days=SMOOTHING_WINDOW * 5)).date()
        df = con.execute(
            "SELECT ts, signal_name, score FROM signals_daily "
            "WHERE ts >= ? AND ts <= ? AND score IS NOT NULL",
            [load_start, ts_end.date()],
        ).df()
        if df.empty:
            return {"start": start.isoformat(), "end": end.isoformat(),
                    "skipped": "no_signals", "n_days": 0}
        df["ts"] = pd.to_datetime(df["ts"])
        wide = df.pivot(index="ts", columns="signal_name", values="score").sort_index()

        # 5-day trailing median of raw scores (the same recipe as
        # `_smoothed_scores` but vectorised across the whole range).
        smoothed = wide.rolling(window=SMOOTHING_WINDOW, min_periods=1).median()

        # Iterate asofs and classify with hysteresis.
        asofs = smoothed.index[(smoothed.index >= ts_start) & (smoothed.index <= ts_end)]
        rows = []
        prior_regime: str | None = None
        days_in_regime: int = 0
        pending_proposed: str | None = None
        pending_days: int = 0

        for ts_asof in asofs:
            smoothed_row = smoothed.loc[ts_asof].dropna().to_dict()
            if not smoothed_row:
                continue
            proposed, conf, comp = classify(smoothed_row)

            if prior_regime is None:
                committed = proposed
                days_in_regime = 1
                pending_proposed = None
                pending_days = 0
            elif proposed == prior_regime:
                committed = prior_regime
                days_in_regime += 1
                pending_proposed = None
                pending_days = 0
            else:
                # A non-matching proposal — apply hysteresis. Track how many
                # consecutive days the same alternative has held; flip when
                # the streak reaches `hyst`.
                if proposed == pending_proposed:
                    pending_days += 1
                else:
                    pending_proposed = proposed
                    pending_days = 1
                if pending_days >= hyst:
                    committed = proposed
                    days_in_regime = 1
                    pending_proposed = None
                    pending_days = 0
                else:
                    committed = prior_regime
                    days_in_regime += 1

            rows.append({
                "ts": ts_asof.date(),
                "regime": committed,
                "prev_regime": prior_regime,
                "confidence": float(conf),
                "days_in_regime": days_in_regime,
                "components": json.dumps(comp),
                "computed_at": datetime.utcnow(),
            })
            prior_regime = committed

        if not rows:
            return {"start": start.isoformat(), "end": end.isoformat(),
                    "skipped": "no_asofs", "n_days": 0}

        # Idempotent overwrite: delete the touched range, then bulk insert.
        ts_first = rows[0]["ts"]
        ts_last  = rows[-1]["ts"]
        con.execute(
            "DELETE FROM regime_history WHERE ts >= ? AND ts <= ?",
            [ts_first, ts_last],
        )
        con.register("incoming_regimes", pd.DataFrame(rows))
        try:
            con.execute(
                "INSERT INTO regime_history "
                "(ts, regime, prev_regime, confidence, days_in_regime, components, computed_at) "
                "SELECT ts, regime, prev_regime, confidence, days_in_regime, components, computed_at "
                "FROM incoming_regimes"
            )
        finally:
            con.unregister("incoming_regimes")

        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "n_days": len(rows),
            "n_transitions": sum(1 for r in rows if r["prev_regime"] and r["prev_regime"] != r["regime"]),
        }


def run_regime_for_date(cfg: Config, asof: date, horizon: str = "daily") -> dict:
    hyst = HYSTERESIS_DAYS.get(horizon, 2)
    with connect(cfg.storage.duckdb_path) as con:
        today_sig = _signals_at(con, asof)
        if not today_sig:
            return {"asof": asof.isoformat(), "skipped": "no_signals"}

        proposed_today, conf_today, comp = _propose_at(con, asof) or (REGIME_UNCERTAIN, 0.30, {})
        prior_regime, prior_days = _load_last_regime(con, asof)

        committed = prior_regime
        days_in_regime = (prior_days or 0) + 1
        if prior_regime is None:
            committed = proposed_today
            days_in_regime = 1
        elif proposed_today == prior_regime:
            committed = prior_regime
        else:
            # Hysteresis: flip only if `hyst` consecutive smoothed proposals
            # (including today) agree on the new regime.
            past_dates = _load_recent_proposed(con, asof, lookback_days=hyst + 1)
            past_dates = [d for d in past_dates if d != pd.Timestamp(asof)]
            past_dates = past_dates[-(hyst - 1):] if hyst > 1 else []
            proposals = []
            for d in past_dates:
                p = _propose_at(con, d.date() if hasattr(d, "date") else d)
                if p:
                    proposals.append(p[0])
            if all(p == proposed_today for p in proposals) and len(proposals) >= max(0, hyst - 1):
                committed = proposed_today
                days_in_regime = 1
            else:
                committed = prior_regime
                days_in_regime = (prior_days or 0) + 1

        con.execute(
            "DELETE FROM regime_history WHERE ts = ?", [asof]
        )
        con.execute(
            "INSERT INTO regime_history (ts, regime, prev_regime, confidence, days_in_regime, components, computed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            [asof, committed, prior_regime, float(conf_today), days_in_regime,
             json.dumps(comp), datetime.utcnow()],
        )

    return {
        "asof": asof.isoformat(),
        "regime": committed,
        "proposed": proposed_today,
        "prev_regime": prior_regime,
        "days_in_regime": days_in_regime,
        "confidence": conf_today,
        "flipped": prior_regime != committed,
    }
