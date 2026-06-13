"""China macro adapter (design note D4).

Pulls PBOC policy + funding rates via akshare/Eastmoney for the HK report's
new "China Policy & Liquidity" subsection. The series stay event-sparse (RRR
moves only when PBOC announces) and daily-dense (SHIBOR) — both share one
long-form table for simplicity:

  china_macro_series (ts, series_id, value, meta_json, ingested_at)
  PK (ts, series_id)

Series captured:
  - RRR_LARGE_BANKS — Required Reserve Ratio for large commercial banks (%),
    via `macro_china_reserve_requirement_ratio`. Event series.
  - LPR_1Y, LPR_5Y — Loan Prime Rates (%), monthly cadence from PBOC,
    via `macro_china_lpr`. Event series.
  - SHIBOR_3M, SHIBOR_1Y — Shanghai Interbank Offered Rate (%), daily,
    via `macro_china_shibor_all`. Funding-cost proxy.

The HK report's §China Policy & Liquidity surfaces the latest value of each
plus the most-recent change date (which is what HK macro readers actually
look at).
"""
from __future__ import annotations

import json
import logging
from datetime import date, datetime
from typing import Callable

import duckdb
import pandas as pd

log = logging.getLogger(__name__)


def _try_import_akshare():
    try:
        import akshare as ak
        return ak
    except ImportError:
        log.info("china_macro: akshare not installed; skipping")
        return None


# ---- RRR ---------------------------------------------------------------

def fetch_rrr() -> pd.DataFrame:
    """Returns [ts, value, magnitude] — ts is the announce-date,
    value is the post-change RRR for large banks, magnitude is the
    percentage-point change. Empty on failure."""
    ak = _try_import_akshare()
    if ak is None:
        return pd.DataFrame()
    try:
        df = ak.macro_china_reserve_requirement_ratio()
    except Exception as exc:
        log.warning("china_macro RRR fetch failed: %s", exc)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    keep = {
        "公布时间":          "ts",
        "大型金融机构-调整后":  "value",
        "大型金融机构-调整幅度": "magnitude",
    }
    df = df[[c for c in keep if c in df.columns]].rename(columns=keep)
    if "ts" not in df.columns:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"], format="%Y年%m月%d日", errors="coerce")
    df = df.dropna(subset=["ts", "value"]).copy()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    if "magnitude" in df.columns:
        df["magnitude"] = pd.to_numeric(df["magnitude"], errors="coerce")
    # GFC-era source can carry two rows for the same announce date (e.g.
    # 2008-06-07). Keep the LAST (largest |magnitude| at the same date is the
    # consolidated post-change level).
    df = df.drop_duplicates(subset=["ts"], keep="last")
    return df.sort_values("ts")


# ---- LPR ---------------------------------------------------------------

def fetch_lpr() -> pd.DataFrame:
    """Returns [ts, lpr_1y, lpr_5y]. Empty on failure."""
    ak = _try_import_akshare()
    if ak is None:
        return pd.DataFrame()
    try:
        df = ak.macro_china_lpr()
    except Exception as exc:
        log.warning("china_macro LPR fetch failed: %s", exc)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"TRADE_DATE": "ts", "LPR1Y": "lpr_1y", "LPR5Y": "lpr_5y"})
    if "ts" not in df.columns:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.dropna(subset=["ts"]).copy()
    df = df[["ts", "lpr_1y", "lpr_5y"]] if {"lpr_1y", "lpr_5y"}.issubset(df.columns) else pd.DataFrame()
    return df.sort_values("ts")


# ---- SHIBOR ------------------------------------------------------------

def fetch_shibor() -> pd.DataFrame:
    """Returns [ts, shibor_3m, shibor_1y] — daily."""
    ak = _try_import_akshare()
    if ak is None:
        return pd.DataFrame()
    try:
        df = ak.macro_china_shibor_all()
    except Exception as exc:
        log.warning("china_macro SHIBOR fetch failed: %s", exc)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"日期": "ts", "3M-定价": "shibor_3m", "1Y-定价": "shibor_1y"})
    if "ts" not in df.columns:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df.dropna(subset=["ts"]).copy()
    for c in ("shibor_3m", "shibor_1y"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    cols = ["ts"] + [c for c in ("shibor_3m", "shibor_1y") if c in df.columns]
    return df[cols].sort_values("ts")


# ---- Storage -----------------------------------------------------------

def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS china_macro_series (
            ts          DATE      NOT NULL,
            series_id   VARCHAR   NOT NULL,
            value       DOUBLE,
            meta_json   VARCHAR,
            ingested_at TIMESTAMP NOT NULL,
            PRIMARY KEY (ts, series_id)
        )
    """)


def _upsert_long(con: duckdb.DuckDBPyConnection,
                  rows: list[dict]) -> int:
    if not rows:
        return 0
    ensure_schema(con)
    df = pd.DataFrame(rows)
    df["ingested_at"] = datetime.utcnow()
    df["ts"] = pd.to_datetime(df["ts"])
    cols = ["ts", "series_id", "value", "meta_json", "ingested_at"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    con.register("incoming_china", df)
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "DELETE FROM china_macro_series WHERE EXISTS "
            "(SELECT 1 FROM incoming_china i WHERE i.ts = china_macro_series.ts "
            "AND i.series_id = china_macro_series.series_id)"
        )
        con.execute("INSERT INTO china_macro_series SELECT * FROM incoming_china")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister("incoming_china")
    return len(df)


def fetch_and_store_all(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Refresh RRR, LPR, SHIBOR. Returns rows-written per source."""
    out: dict[str, int] = {}

    # RRR — event series. Store value AND magnitude in meta_json for surfacing.
    rrr = fetch_rrr()
    if not rrr.empty:
        rows = [
            {"ts": r.ts.date(), "series_id": "RRR_LARGE_BANKS",
             "value": float(r.value),
             "meta_json": json.dumps({"magnitude": float(r.magnitude)
                                       if not pd.isna(r.magnitude) else None})}
            for r in rrr.itertuples()
        ]
        out["rrr"] = _upsert_long(con, rows)
    else:
        out["rrr"] = 0

    # LPR — monthly cadence. Store as two series.
    lpr = fetch_lpr()
    if not lpr.empty:
        rows = []
        for r in lpr.itertuples():
            if not pd.isna(r.lpr_1y):
                rows.append({"ts": r.ts.date(), "series_id": "LPR_1Y",
                             "value": float(r.lpr_1y)})
            if not pd.isna(r.lpr_5y):
                rows.append({"ts": r.ts.date(), "series_id": "LPR_5Y",
                             "value": float(r.lpr_5y)})
        out["lpr"] = _upsert_long(con, rows)
    else:
        out["lpr"] = 0

    # SHIBOR — daily. Two tenors.
    sb = fetch_shibor()
    if not sb.empty:
        rows = []
        for r in sb.itertuples():
            if "shibor_3m" in sb.columns and not pd.isna(r.shibor_3m):
                rows.append({"ts": r.ts.date(), "series_id": "SHIBOR_3M",
                             "value": float(r.shibor_3m)})
            if "shibor_1y" in sb.columns and not pd.isna(r.shibor_1y):
                rows.append({"ts": r.ts.date(), "series_id": "SHIBOR_1Y",
                             "value": float(r.shibor_1y)})
        out["shibor"] = _upsert_long(con, rows)
    else:
        out["shibor"] = 0

    return out


# ---- Read helpers ------------------------------------------------------

def latest_value(con: duckdb.DuckDBPyConnection, series_id: str,
                 asof: date | None = None) -> dict | None:
    """Returns {ts, value, magnitude} for the most recent observation of
    `series_id` at or before `asof` (default: today)."""
    asof_str = (asof or date.today()).isoformat()
    try:
        row = con.execute(
            "SELECT ts, value, meta_json FROM china_macro_series "
            "WHERE series_id = ? AND ts <= ? ORDER BY ts DESC LIMIT 1",
            [series_id, asof_str],
        ).fetchone()
    except duckdb.CatalogException:
        return None
    if not row:
        return None
    out = {"ts": row[0], "value": row[1]}
    if row[2]:
        try:
            out.update(json.loads(row[2]))
        except Exception:
            pass
    return out
