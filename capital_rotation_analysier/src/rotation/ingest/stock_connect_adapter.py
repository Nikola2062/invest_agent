"""Stock Connect flow adapter (design notes D2 + D3).

Pulls Stock Connect data via akshare/Eastmoney.

D2 — aggregate daily flows (source: stock_hsgt_hist_em):
  - Southbound (港股通南向资金) — Mainland investors buying/selling HK stocks.
    Last row 2026-06-10 — LIVE, daily, the actionable signal for the HK report.
  - Northbound (北向资金) — HK/foreign investors buying/selling A-shares.
    Last row 2024-08-16 — Chinese authorities discontinued public daily net-flow
    publication after that date. We keep the history we have, mark the series
    frozen, and do not retry — that's a sourcing reality, not a bug.
  Units: 100M CNY (亿). Stored in `stock_connect_flows` keyed by (ts, direction).

D3 — per-HK-stock Southbound holdings (source: stock_hsgt_stock_statistics_em):
  Mainland holdings of every HK Stock-Connect-eligible name, with 1d/5d/10d
  market-value delta. Direct view into which HK tickers Mainland money is
  accumulating vs distributing. Stored in `stock_connect_holdings` keyed by
  (ts, symbol). The HK report joins this against its 15-name universe to add
  a SB-flow column to §13/14 and a Top SB Buyers/Sellers table.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Literal

import duckdb
import pandas as pd

log = logging.getLogger(__name__)


_SYMBOL_FOR = {
    "southbound": "南向资金",   # Mainland -> HK
    "northbound": "北向资金",   # HK/foreign -> A-shares (frozen since 2024-08-16)
}

Direction = Literal["southbound", "northbound"]


def fetch_history(direction: Direction) -> pd.DataFrame:
    """Returns columns [ts, net_buy_cny_100m, hist_cum_cny_100m, holding_value_cny].
    Empty if akshare/Eastmoney is unreachable. Drops rows where the net-buy
    column is null (Eastmoney pads recent calendar rows with NaN on weekends/
    after the publishing cutoff)."""
    try:
        import akshare as ak
    except ImportError:
        log.info("stock_connect: akshare not installed; skipping %s", direction)
        return pd.DataFrame()

    sym = _SYMBOL_FOR[direction]
    try:
        df = ak.stock_hsgt_hist_em(symbol=sym)
    except Exception as exc:
        log.warning("stock_connect: akshare fetch failed for %s: %s", direction, exc)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    keep = {
        "日期":      "ts",
        "当日成交净买额":  "net_buy_cny_100m",
        "历史累计净买额":  "hist_cum_cny_100m",
        "持股市值":         "holding_value_cny",
    }
    df = df[[c for c in keep if c in df.columns]].rename(columns=keep)
    if "ts" not in df.columns or "net_buy_cny_100m" not in df.columns:
        log.warning("stock_connect: unexpected column shape for %s: %s",
                    direction, list(df.columns))
        return pd.DataFrame()
    df = df.dropna(subset=["net_buy_cny_100m"]).copy()
    df["ts"] = pd.to_datetime(df["ts"]).dt.date
    for c in ("net_buy_cny_100m", "hist_cum_cny_100m", "holding_value_cny"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS stock_connect_flows (
            ts                  DATE      NOT NULL,
            direction           VARCHAR   NOT NULL,
            net_buy_cny_100m    DOUBLE,
            hist_cum_cny_100m   DOUBLE,
            holding_value_cny   DOUBLE,
            ingested_at         TIMESTAMP NOT NULL,
            PRIMARY KEY (ts, direction)
        )
    """)


def upsert(con: duckdb.DuckDBPyConnection, direction: Direction,
           df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    ensure_schema(con)
    df = df.copy()
    df["direction"] = direction
    df["ingested_at"] = datetime.utcnow()
    df["ts"] = pd.to_datetime(df["ts"])
    cols = ["ts", "direction", "net_buy_cny_100m", "hist_cum_cny_100m",
            "holding_value_cny", "ingested_at"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    con.register("incoming_sc", df)
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "DELETE FROM stock_connect_flows WHERE EXISTS "
            "(SELECT 1 FROM incoming_sc i WHERE i.ts = stock_connect_flows.ts "
            "AND i.direction = stock_connect_flows.direction)"
        )
        con.execute("INSERT INTO stock_connect_flows SELECT * FROM incoming_sc")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister("incoming_sc")
    return len(df)


def fetch_and_store_all(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Refresh Southbound (live) and Northbound (frozen since 2024-08-16)."""
    out: dict[str, int] = {}
    for direction in ("southbound", "northbound"):
        df = fetch_history(direction)  # type: ignore[arg-type]
        out[direction] = upsert(con, direction, df)  # type: ignore[arg-type]
    return out


# ============================================================
# D3 — per-stock Southbound holdings
# ============================================================

def _hk_code_5digit(symbol_dot_hk: str) -> str:
    """Normalize '0700.HK' / '700.HK' -> '00700' (Eastmoney's 5-digit format)."""
    code = symbol_dot_hk.split(".")[0]
    return code.zfill(5)


def fetch_holdings(start_date: date, end_date: date) -> pd.DataFrame:
    """Per-stock Southbound holdings for [start_date, end_date].

    Returns columns:
      ts, symbol (HK 5-digit), name, close_hkd, daily_pct,
      shares_held, market_value_hkd, pct_of_shares_outstanding,
      mv_chg_1d_hkd, mv_chg_5d_hkd, mv_chg_10d_hkd.

    Empty if akshare/Eastmoney is unreachable. The endpoint paginates per day
    so wide windows are slow (a 30-day pull is ~30 HTTP calls)."""
    try:
        import akshare as ak
    except ImportError:
        log.info("stock_connect: akshare not installed; skipping holdings")
        return pd.DataFrame()

    try:
        df = ak.stock_hsgt_stock_statistics_em(
            symbol="南向持股",
            start_date=start_date.strftime("%Y%m%d"),
            end_date=end_date.strftime("%Y%m%d"),
        )
    except Exception as exc:
        log.warning("stock_connect: holdings fetch failed: %s", exc)
        return pd.DataFrame()
    if df is None or df.empty:
        return pd.DataFrame()

    keep = {
        "持股日期":            "ts",
        "股票代码":            "symbol",
        "股票简称":            "name",
        "当日收盘价":          "close_hkd",
        "当日涨跌幅":          "daily_pct",
        "持股数量":            "shares_held",
        "持股市值":            "market_value_hkd",
        "持股数量占发行股百分比": "pct_of_shares_outstanding",
        "持股市值变化-1日":     "mv_chg_1d_hkd",
        "持股市值变化-5日":     "mv_chg_5d_hkd",
        "持股市值变化-10日":    "mv_chg_10d_hkd",
    }
    df = df[[c for c in keep if c in df.columns]].rename(columns=keep)
    if "ts" not in df.columns or "symbol" not in df.columns:
        log.warning("stock_connect: unexpected holdings shape: %s", list(df.columns))
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"]).dt.date
    df["symbol"] = df["symbol"].astype(str).str.zfill(5)
    for c in ("close_hkd", "daily_pct", "shares_held", "market_value_hkd",
              "pct_of_shares_outstanding",
              "mv_chg_1d_hkd", "mv_chg_5d_hkd", "mv_chg_10d_hkd"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df


def ensure_holdings_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("""
        CREATE TABLE IF NOT EXISTS stock_connect_holdings (
            ts                            DATE     NOT NULL,
            symbol                        VARCHAR  NOT NULL,
            name                          VARCHAR,
            close_hkd                     DOUBLE,
            daily_pct                     DOUBLE,
            shares_held                   DOUBLE,
            market_value_hkd              DOUBLE,
            pct_of_shares_outstanding     DOUBLE,
            mv_chg_1d_hkd                 DOUBLE,
            mv_chg_5d_hkd                 DOUBLE,
            mv_chg_10d_hkd                DOUBLE,
            ingested_at                   TIMESTAMP NOT NULL,
            PRIMARY KEY (ts, symbol)
        )
    """)


def upsert_holdings(con: duckdb.DuckDBPyConnection, df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    ensure_holdings_schema(con)
    df = df.copy()
    df["ingested_at"] = datetime.utcnow()
    df["ts"] = pd.to_datetime(df["ts"])
    cols = ["ts", "symbol", "name", "close_hkd", "daily_pct",
            "shares_held", "market_value_hkd", "pct_of_shares_outstanding",
            "mv_chg_1d_hkd", "mv_chg_5d_hkd", "mv_chg_10d_hkd", "ingested_at"]
    for c in cols:
        if c not in df.columns:
            df[c] = None
    df = df[cols]
    con.register("incoming_sch", df)
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "DELETE FROM stock_connect_holdings WHERE EXISTS "
            "(SELECT 1 FROM incoming_sch i WHERE i.ts = stock_connect_holdings.ts "
            "AND i.symbol = stock_connect_holdings.symbol)"
        )
        con.execute("INSERT INTO stock_connect_holdings SELECT * FROM incoming_sch")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister("incoming_sch")
    return len(df)


def refresh_holdings(con: duckdb.DuckDBPyConnection, asof: date,
                     lookback_days: int = 7) -> int:
    """Daily snapshot of the per-stock SB holdings panel.

    A lookback of 7 days keeps each daily refresh cheap (~7 HTTP calls) while
    still catching last week's data if the prior run was skipped. Idempotent
    by (ts, symbol)."""
    start = asof - timedelta(days=lookback_days)
    df = fetch_holdings(start, asof)
    return upsert_holdings(con, df)


def load_holdings_for_universe(con: duckdb.DuckDBPyConnection,
                                symbols: list[str], asof: date,
                                lookback_days: int = 30) -> pd.DataFrame:
    """Get SB holdings rows for our HK universe.

    `symbols` are config-style ('0700.HK'); we map to Eastmoney's 5-digit form
    and join in the DB. Returns long-form (one row per (ts, symbol)) filtered
    to the request universe and the trailing `lookback_days` sessions."""
    if not symbols:
        return pd.DataFrame()
    em_codes = [_hk_code_5digit(s) for s in symbols]
    start = (pd.Timestamp(asof) - pd.Timedelta(days=lookback_days)).date()
    placeholders = ",".join(["?"] * len(em_codes))
    try:
        df = con.execute(
            f"""
            SELECT ts, symbol, name, close_hkd, daily_pct,
                   shares_held, market_value_hkd, pct_of_shares_outstanding,
                   mv_chg_1d_hkd, mv_chg_5d_hkd, mv_chg_10d_hkd
            FROM stock_connect_holdings
            WHERE ts BETWEEN ? AND ? AND symbol IN ({placeholders})
            ORDER BY ts, symbol
            """,
            [start, asof, *em_codes],
        ).df()
    except duckdb.CatalogException:
        return pd.DataFrame()
    if df.empty:
        return df
    df["ts"] = pd.to_datetime(df["ts"])
    # Map back to config form by adding a `symbol_dot_hk` column for convenience.
    em_to_dot = {_hk_code_5digit(s): s for s in symbols}
    df["symbol_dot_hk"] = df["symbol"].map(em_to_dot)
    return df


def load_panel(con: duckdb.DuckDBPyConnection, asof: date,
               lookback_days: int = 90) -> pd.DataFrame:
    """Wide panel: index=date, columns=direction, values=net_buy_cny_100m."""
    start = (pd.Timestamp(asof) - pd.Timedelta(days=lookback_days)).date()
    df = con.execute(
        "SELECT ts, direction, net_buy_cny_100m FROM stock_connect_flows "
        "WHERE ts BETWEEN ? AND ? ORDER BY ts",
        [start, asof],
    ).df()
    if df.empty:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"])
    return df.pivot(index="ts", columns="direction",
                    values="net_buy_cny_100m").sort_index()
