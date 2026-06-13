"""Glue layer: read bars from store, compute metrics + signals, write back.

Separates pure math (metrics.py, signals.py) from persistence (store.py).
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Iterable

import duckdb
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from . import metrics as M
from . import signals as S
from .ingest.fred_adapter import load_fred_panel
from .store import connect
from .config import Config

log = logging.getLogger(__name__)


def load_panel(
    con: duckdb.DuckDBPyConnection,
    end_date: date,
    lookback_days: int = 400,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (adj_close_wide, volume_wide) DataFrames indexed by date.

    HKEX-listed assets (`equity_hk`) are excluded — they trade on a different
    calendar and would otherwise produce mostly-NaN rows in the NYSE-aligned
    panel. The Greater China Holdings report section reads them separately.
    """
    df = con.execute(
        """
        SELECT ts, symbol, adj_close, volume
        FROM raw_bars
        WHERE ts >= ? AND ts <= ?
          AND asset_class != 'equity_hk'
        """,
        [pd.Timestamp(end_date) - pd.Timedelta(days=lookback_days), pd.Timestamp(end_date)],
    ).df()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"])
    close = df.pivot(index="ts", columns="symbol", values="adj_close").sort_index()
    volume = df.pivot(index="ts", columns="symbol", values="volume").sort_index()

    # Align to NYSE trading days. yfinance returns BTC/ETH on weekends, which
    # otherwise leaves equity rows sparse (~38% valid in a 252d window) and
    # breaks the robust-z normalization for everything except crypto. Crypto
    # weekend bars still live in raw_bars; they just don't drive daily signals.
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=close.index.min(), end_date=close.index.max())
    nyse_days = pd.DatetimeIndex(pd.to_datetime(sched.index).normalize())
    keep = close.index.intersection(nyse_days)
    return close.loc[keep], volume.loc[keep]


def compute_metrics_for(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    asof: pd.Timestamp,
) -> pd.DataFrame:
    """Returns one row per symbol with all per-asset metrics on `asof`."""
    if asof not in close.index:
        return pd.DataFrame()

    r_d = M.log_returns(close, 1)
    r_w = M.log_returns(close, 5)
    r_m = M.log_returns(close, 21)
    r_q = M.log_returns(close, 63)
    vol_30 = M.realized_vol(close, 30)
    vol_ratio = M.vol_of_vol_ratio(close)
    rv = M.relative_volume(volume)
    vz = M.volume_zscore(volume)
    rs = M.rs_rank(close)
    rsc1 = M.rank_change(rs, 1)
    rsc5 = M.rank_change(rs, 5)
    rsc21 = M.rank_change(rs, 21)
    # Δ²RS over 5d: change in 5d-velocity vs 5d ago. Positive = accelerating
    # into leadership, negative = decelerating. The earliest leading signal.
    rs_accel_5 = rsc5 - rsc5.shift(5)

    syms = close.columns
    out = pd.DataFrame(index=syms)
    out["r_d"]     = r_d.loc[asof] if asof in r_d.index else np.nan
    out["r_w"]     = r_w.loc[asof] if asof in r_w.index else np.nan
    out["r_m"]     = r_m.loc[asof] if asof in r_m.index else np.nan
    out["r_q"]     = r_q.loc[asof] if asof in r_q.index else np.nan
    out["vol_30"]  = vol_30.loc[asof] if asof in vol_30.index else np.nan
    out["vol_ratio"] = vol_ratio.loc[asof] if asof in vol_ratio.index else np.nan
    out["rv"]      = rv.loc[asof] if asof in rv.index else np.nan
    out["vz"]      = vz.loc[asof] if asof in vz.index else np.nan
    out["rs_rank"]      = rs.loc[asof] if asof in rs.index else pd.NA
    out["rs_change_1"]  = rsc1.loc[asof] if asof in rsc1.index else pd.NA
    out["rs_change_5"]  = rsc5.loc[asof] if asof in rsc5.index else pd.NA
    out["rs_change_21"] = rsc21.loc[asof] if asof in rsc21.index else pd.NA
    out["rs_accel_5"]   = rs_accel_5.loc[asof] if asof in rs_accel_5.index else pd.NA

    out = out.reset_index().rename(columns={"index": "symbol", "symbol": "symbol"})
    out.columns.name = None
    return out


def upsert_metrics(con: duckdb.DuckDBPyConnection, ts: pd.Timestamp, rows: pd.DataFrame) -> int:
    if rows.empty:
        return 0
    rows = rows.copy()
    rows["ts"] = ts.date() if hasattr(ts, "date") else ts
    rows["computed_at"] = datetime.utcnow()
    # Coerce rs_* nullable Int to plain (None for NA)
    for c in ("rs_rank", "rs_change_1", "rs_change_5", "rs_change_21", "rs_accel_5"):
        rows[c] = rows[c].astype("object").where(rows[c].notna(), None)

    write_cols = ["ts", "symbol", "r_d", "r_w", "r_m", "r_q", "vol_30", "vol_ratio",
                  "rv", "vz", "rs_rank", "rs_change_1", "rs_change_5", "rs_change_21",
                  "rs_accel_5", "computed_at"]
    rows = rows[write_cols]
    con.register("incoming_metrics", rows)
    try:
        con.execute(
            """
            DELETE FROM metrics_daily
            WHERE EXISTS (SELECT 1 FROM incoming_metrics i
                          WHERE i.ts = metrics_daily.ts AND i.symbol = metrics_daily.symbol)
            """
        )
        # Explicit (cols) (cols) so this survives ALTER TABLE ADD COLUMN, which
        # appends at the end and would break a positional SELECT * pattern.
        col_list = ", ".join(write_cols)
        con.execute(
            f"INSERT INTO metrics_daily ({col_list}) SELECT {col_list} FROM incoming_metrics"
        )
    finally:
        con.unregister("incoming_metrics")
    return len(rows)


def compute_signals_for(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    asof: pd.Timestamp,
    fred: pd.DataFrame | None = None,
) -> dict[str, dict]:
    return S.compute_all(close, volume, asof, fred=fred)


def upsert_signals(
    con: duckdb.DuckDBPyConnection,
    ts: pd.Timestamp,
    results: dict[str, dict],
) -> int:
    ts_date = ts.date() if hasattr(ts, "date") else ts
    rows = []
    now = datetime.utcnow()
    for name, payload in results.items():
        rows.append({
            "ts": ts_date,
            "signal_name": name,
            "score": payload.get("score"),
            "confidence": payload.get("confidence"),
            "components": json.dumps(_jsonable(payload)),
            "computed_at": now,
        })
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    con.register("incoming_sigs", df)
    try:
        con.execute(
            """
            DELETE FROM signals_daily
            WHERE EXISTS (SELECT 1 FROM incoming_sigs i
                          WHERE i.ts = signals_daily.ts AND i.signal_name = signals_daily.signal_name)
            """
        )
        con.execute(
            "INSERT INTO signals_daily (ts, signal_name, score, confidence, components, computed_at) "
            "SELECT ts, signal_name, score, confidence, components, computed_at FROM incoming_sigs"
        )
    finally:
        con.unregister("incoming_sigs")
    return len(rows)


def _jsonable(obj):
    """Recursively coerce numpy / pandas scalars into plain Python for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(x) for x in obj]
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


class _PanelCache:
    """Memoize the panel-wide outputs of metrics.* across many asof iterations.

    Each metric function in `metrics.py` is pure over its inputs. During a
    backfill the inputs (the wide `close` / `volume` panels) don't change —
    yet each call rebuilds a 2500×32 DataFrame of returns / robust-z / vol.
    Repeating that 2500 times is what makes a naive backfill loop O(N²).

    This wrapper installs cached versions of the hot metric functions for
    the lifetime of a `with` block, restoring the originals on exit. The
    cache key is `(id(arg0), other_args)`; because the panel reference is
    stable across the loop, every call after the first is an O(1) lookup.
    """

    HOT_FUNCS = ("log_returns", "robust_z", "realized_vol", "vol_of_vol_ratio",
                 "relative_volume", "volume_zscore", "rs_rank", "rank_change",
                 "blended_return")

    def __init__(self):
        self._originals: dict = {}
        self._cache: dict = {}

    def __enter__(self):
        from . import metrics as M
        self._originals = {}
        for name in self.HOT_FUNCS:
            orig = getattr(M, name)
            self._originals[name] = orig
            setattr(M, name, self._wrap(name, orig))
        return self

    def __exit__(self, exc_type, exc, tb):
        from . import metrics as M
        for name, orig in self._originals.items():
            setattr(M, name, orig)
        self._cache.clear()

    def _wrap(self, name, orig):
        cache = self._cache
        def wrapped(*args, **kwargs):
            key = _content_key(name, args, kwargs)
            if key is None:                       # unhashable -> bypass
                return orig(*args, **kwargs)
            hit = cache.get(key)
            if hit is not None:
                return hit
            out = orig(*args, **kwargs)
            cache[key] = out
            return out
        wrapped.__wrapped__ = orig
        return wrapped


def _content_key(fname: str, args: tuple, kwargs: dict):
    """Cheap content-aware cache key for the hot metric functions.

    Series/DataFrame instances get keyed on (name, len, first-value, last-value)
    instead of `id()` because pandas returns a fresh Series wrapper on every
    `df["col"]` access — `id()` would never hit. The pseudo-hash is bound to
    the panel (which is frozen during the backfill) and to the column name,
    so collisions across calls inside one backfill are vanishingly rare.

    Returns None if the kwargs contain something unhashable; the caller then
    bypasses the cache for that call.
    """
    parts: list = [fname]
    for a in args:
        parts.append(_arg_token(a))
    for k in sorted(kwargs):
        parts.append((k, _arg_token(kwargs[k])))
    try:
        return tuple(parts)
    except TypeError:
        return None


def _arg_token(a):
    if isinstance(a, pd.Series):
        try:
            return ("S", a.name, len(a),
                    float(a.iloc[0]) if len(a) and pd.notna(a.iloc[0]) else None,
                    float(a.iloc[-1]) if len(a) and pd.notna(a.iloc[-1]) else None)
        except Exception:
            return ("S", a.name, len(a))
    if isinstance(a, pd.DataFrame):
        return ("D", id(a))   # panel is stable during backfill
    if isinstance(a, (int, float, str, bool, type(None))):
        return a
    if isinstance(a, dict):
        return tuple(sorted((k, _arg_token(v)) for k, v in a.items()))
    if isinstance(a, (list, tuple)):
        return tuple(_arg_token(x) for x in a)
    return id(a)


def backfill_signals(
    cfg: Config,
    start: date,
    end: date,
) -> dict:
    """Batch recompute of metrics + signals across a date range.

    Loads the bar panel ONCE for [start - 400d, end], then iterates each NYSE
    trading day in [start, end], computing metrics + signals against the same
    in-memory panel. A `_PanelCache` memoizes panel-wide rolling-z / log-return
    work across asof iterations — without it, the per-day compute is O(N²) in
    the panel size and a 10-year backfill is hours; with it, it is seconds.

    Returns a summary dict with counts; per-day signal scores are NOT echoed
    to keep the return value small.
    """
    ts_start = pd.Timestamp(start)
    ts_end = pd.Timestamp(end)
    with connect(cfg.storage.duckdb_path) as con:
        # Load panel with enough lookback to satisfy the 252d robust-z window
        # plus a buffer for the warm-up of the chained rolling computations.
        load_start = (ts_start - pd.Timedelta(days=400)).date()
        df = con.execute(
            "SELECT ts, symbol, adj_close, volume FROM raw_bars "
            "WHERE ts >= ? AND ts <= ? AND asset_class != 'equity_hk'",
            [load_start, ts_end.date()],
        ).df()
        if df.empty:
            return {"start": start.isoformat(), "end": end.isoformat(),
                    "skipped": "no_data", "n_days": 0}
        df["ts"] = pd.to_datetime(df["ts"])
        close = df.pivot(index="ts", columns="symbol", values="adj_close").sort_index()
        volume = df.pivot(index="ts", columns="symbol", values="volume").sort_index()
        import pandas_market_calendars as mcal
        nyse = mcal.get_calendar("NYSE")
        sched = nyse.schedule(start_date=close.index.min(), end_date=close.index.max())
        nyse_days = pd.DatetimeIndex(pd.to_datetime(sched.index).normalize())
        keep = close.index.intersection(nyse_days)
        close = close.loc[keep]
        volume = volume.loc[keep]

        try:
            fred_panel = load_fred_panel(con, end)
        except duckdb.CatalogException:
            fred_panel = None  # fred_series table not created yet (FRED opt-in)

        asofs = close.index[(close.index >= ts_start) & (close.index <= ts_end)]
        n_metric_rows = 0
        n_signal_rows = 0
        n_days_processed = 0
        with _PanelCache():
            for asof_ts in asofs:
                metrics = compute_metrics_for(close, volume, asof_ts)
                if metrics.empty:
                    continue
                n_metric_rows += upsert_metrics(con, asof_ts, metrics)
                sigs = compute_signals_for(close, volume, asof_ts, fred=fred_panel)
                n_signal_rows += upsert_signals(con, asof_ts, sigs)
                n_days_processed += 1

        return {
            "start": start.isoformat(),
            "end": end.isoformat(),
            "n_days": n_days_processed,
            "n_metric_rows": n_metric_rows,
            "n_signal_rows": n_signal_rows,
            "n_symbols": int(close.shape[1]),
        }


def run_signals_for_date(cfg: Config, asof: date) -> dict:
    """End-to-end: load -> compute -> persist."""
    ts = pd.Timestamp(asof)
    with connect(cfg.storage.duckdb_path) as con:
        close, volume = load_panel(con, asof)
        if close.empty:
            return {"asof": asof.isoformat(), "n_symbols": 0, "n_signals": 0, "skipped": "no_data"}

        # Snap asof to the most recent date in the panel (handles weekends/holidays).
        if ts not in close.index:
            cal_dates = close.index[close.index <= ts]
            if len(cal_dates) == 0:
                return {"asof": asof.isoformat(), "n_symbols": 0, "n_signals": 0, "skipped": "no_history"}
            ts = cal_dates[-1]

        metrics = compute_metrics_for(close, volume, ts)
        n_metrics = upsert_metrics(con, ts, metrics)
        try:
            fred_panel = load_fred_panel(con, asof)
        except duckdb.CatalogException:
            fred_panel = None  # fred_series table not created yet (FRED opt-in)
        sigs = compute_signals_for(close, volume, ts, fred=fred_panel)
        n_signals = upsert_signals(con, ts, sigs)

        return {
            "asof": ts.date().isoformat(),
            "n_symbols": int(close.shape[1]),
            "n_metric_rows": n_metrics,
            "n_signals": n_signals,
            "signals_summary": {
                name: {"score": p.get("score"), "confidence": p.get("confidence")}
                for name, p in sigs.items()
            },
        }
