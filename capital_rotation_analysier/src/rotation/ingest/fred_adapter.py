"""FRED (Federal Reserve Economic Data) adapter — daily macro series.

Fetches macro series that feed the Inflation, Liquidity and Recession signals
(see design note D1). Series fall into three groups:

  Inflation expectations (signed contributors to Inflation):
    - T10YIE — 10-yr breakeven inflation (market-implied, beats commodity proxy)
    - T5YIFR — 5y5y forward breakeven (Fed's preferred long-run gauge)
    - DFII10 — 10-yr TIPS real yield (crosscheck for the gold-vs-breakeven trade)

  Net Fed liquidity decomposition (contributors to Liquidity composite):
    - WALCL — Fed total assets (weekly, ffill'd to daily)
    - WTREGEN — Treasury General Account (drained TGA = more market liquidity)
    - RRPONTSYD — overnight reverse repo balance (parked cash = less liquidity)
    Net liquidity = WALCL - WTREGEN - RRPONTSYD

  Rates / curve / credit (contributors to Recession):
    - DGS10, DGS2, T10Y2Y — Treasury yields + spread
    - BAMLH0A0HYM2 — ICE BofA HY OAS (true credit spread vs HYG/LQD proxy)

Free API, 120 req/min, key from https://fred.stlouisfed.org/docs/api/api_key.html.
Set FRED_API_KEY env var or leave unset to silently disable.

Stores in a wide `fred_series` table keyed by (series_id, ts).
"""
from __future__ import annotations

import json
import logging
import os
import time
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta
from typing import Iterable

import duckdb
import pandas as pd

log = logging.getLogger(__name__)


FRED_BASE = "https://api.stlouisfed.org/fred/series/observations"

# series_id -> human description (used only in logs/reports).
# Order matches the docstring groupings (inflation / liquidity / rates+credit).
SERIES = {
    # Inflation expectations
    "T10YIE":    "10-Year Breakeven Inflation Rate (daily)",
    "T5YIFR":    "5y5y Forward Breakeven Inflation (daily)",
    "DFII10":    "10-Year TIPS Real Yield (daily)",
    # Net Fed liquidity decomposition
    "WALCL":     "Fed Total Assets, $M (weekly Wed)",
    "WTREGEN":   "Treasury General Account, $M (weekly Wed)",
    "RRPONTSYD": "Overnight Reverse Repurchase Agreements, $B (daily)",
    # Rates / curve / credit
    "DGS10":     "10-Year Treasury Constant Maturity Rate (daily)",
    "DGS2":      "2-Year Treasury Constant Maturity Rate (daily)",
    "T10Y2Y":    "10y-2y Treasury spread (daily)",
    "BAMLH0A0HYM2": "ICE BofA US HY Index OAS (daily)",
    # Policy rate (W8 macro context) — FOMC target upper bound, steps at meetings.
    "DFEDTARU":  "Federal Funds Target Range Upper Limit (daily)",
}


def _have_key() -> bool:
    return bool(os.environ.get("FRED_API_KEY"))


def fetch_series(series_id: str, start: date, end: date | None = None) -> pd.DataFrame:
    """Returns a DataFrame with columns [ts, value]. Empty if no key or no data."""
    key = os.environ.get("FRED_API_KEY")
    if not key:
        log.info("FRED: no API key in env; skipping %s", series_id)
        return pd.DataFrame(columns=["ts", "value"])

    end = end or date.today()
    params = urllib.parse.urlencode({
        "series_id": series_id, "api_key": key, "file_type": "json",
        "observation_start": start.isoformat(),
        "observation_end": end.isoformat(),
    })
    url = f"{FRED_BASE}?{params}"

    for attempt in range(3):
        try:
            with urllib.request.urlopen(url, timeout=15) as resp:
                data = json.loads(resp.read().decode())
            break
        except Exception as exc:
            wait = [1, 4, 16][attempt]
            log.warning("FRED fetch %s attempt %d failed: %s; sleep %ds",
                        series_id, attempt + 1, exc, wait)
            if attempt == 2:
                return pd.DataFrame(columns=["ts", "value"])
            time.sleep(wait)

    rows = data.get("observations", [])
    if not rows:
        return pd.DataFrame(columns=["ts", "value"])
    df = pd.DataFrame(rows)[["date", "value"]]
    df = df.rename(columns={"date": "ts"})
    df["ts"] = pd.to_datetime(df["ts"]).dt.date
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df.dropna(subset=["value"])


def ensure_fred_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS fred_series (
            series_id   VARCHAR NOT NULL,
            ts          DATE    NOT NULL,
            value       DOUBLE  NOT NULL,
            ingested_at TIMESTAMP NOT NULL,
            PRIMARY KEY (series_id, ts)
        )
    """)


def upsert_fred(con: duckdb.DuckDBPyConnection, series_id: str, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    ensure_fred_schema(con)
    df = df.copy()
    df["series_id"] = series_id
    df["ingested_at"] = datetime.utcnow()
    df["ts"] = pd.to_datetime(df["ts"])
    df = df[["series_id", "ts", "value", "ingested_at"]]
    con.register("incoming_fred", df)
    # Atomic DELETE+INSERT — same rationale as store.upsert_bars.
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "DELETE FROM fred_series WHERE EXISTS "
            "(SELECT 1 FROM incoming_fred i WHERE i.series_id = fred_series.series_id AND i.ts = fred_series.ts)"
        )
        con.execute("INSERT INTO fred_series SELECT * FROM incoming_fred")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister("incoming_fred")
    return len(df)


# Default backfill start for FRED series; override via env without a code edit.
def _default_start() -> date:
    return date.fromisoformat(os.environ.get("FRED_START_DATE", "2024-11-01"))


def fetch_and_store_all(
    con: duckdb.DuckDBPyConnection,
    start: date | None = None,
    end: date | None = None,
) -> dict[str, int]:
    """Refresh all configured FRED series. Idempotent. Returns rows written per series."""
    out: dict[str, int] = {}
    if not _have_key():
        log.info("FRED: no API key; skipping all series")
        return out
    start = start or _default_start()
    for sid in SERIES:
        df = fetch_series(sid, start, end)
        out[sid] = upsert_fred(con, sid, df)
    return out


def load_fred_panel(con: duckdb.DuckDBPyConnection, end: date) -> pd.DataFrame:
    """Wide panel indexed by date, columns = FRED series_ids. Empty if no data."""
    df = con.execute(
        "SELECT ts, series_id, value FROM fred_series WHERE ts <= ? ORDER BY ts",
        [end],
    ).df()
    if df.empty:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"])
    return df.pivot(index="ts", columns="series_id", values="value").sort_index()
