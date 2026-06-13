"""ETF flow ingest per the project docs §2.5 / §3.5.

Two methods (per spec):

  **Method B (default, universal)** — AUM-delta proxy. We fetch
  `Ticker.info['sharesOutstanding']` via yfinance daily, compute
  `net_flow_estimate_t = (shares_out_t − shares_out_{t-1}) × vwap_t`, and tag
  the row with `proxy_method='shares_delta'`, `confidence=0.6`. Universal
  coverage of every ETF in our universe. T+0 (same evening as close).

  **Method A (priority enrichment, deferred)** — Issuer-direct CSVs. The 6
  ETFs selected for first wiring when method A is implemented are
  SPY, QQQ, IWM, GLD, TLT, HYG — they cover broad US equity, growth, small-cap,
  gold, duration, credit. Each issuer (SSGA / Invesco / iShares) has a unique
  URL pattern; building those adapters requires per-issuer URL discovery and
  test that the spec author deferred from MVP. The dispatcher already accepts
  `proxy_method='issuer_direct'` so adding method A only requires implementing
  the issuer-specific fetchers; no schema or downstream change needed.

Honest caveats for method B:
  - Yahoo's `sharesOutstanding` is point-in-time, not a date-stamped series.
    We snapshot it on each run; the delta series is built from those snapshots.
    Stale Yahoo snapshots can produce spurious zero-flow days.
  - VWAP is approximated as close (yfinance doesn't expose true VWAP).
  - Net-flow magnitude is therefore order-of-magnitude correct, not precise.
    That's fine for the §3.5 flow_z normalization which only cares about the
    z-score of changes, not absolute dollars.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Iterable

import duckdb
import pandas as pd
import yfinance as yf

from ..config import Config
from .flows_issuer_direct import (
    PRIORITY_ETFS as PRIORITY_ISSUER_DIRECT,
    fetch_issuer_direct,
    reconcile_net_flow,
)

log = logging.getLogger(__name__)

# Method B provenance weight (the project docs §1.6.1 decision 13) — single place to tune.
CONFIDENCE_SHARES_DELTA = 0.60

# HK-listed index ETFs. Individual HK stocks share asset_class=equity_hk in
# config.yaml; we whitelist the actual ETFs explicitly so we don't try to apply
# Method B to single names (where yf.info shares outstanding aren't comparable
# to ETF creation/redemption flows).
HK_ETF_SYMBOLS = {"2800.HK", "2828.HK"}


def fetch_shares_outstanding(symbols: Iterable[str]) -> dict[str, float]:
    """Snapshot yfinance's current shares_outstanding for each ETF.

    Tries in order:
      1. `Ticker.info['sharesOutstanding']` (canonical)
      2. `Ticker.info['shares']` (older field name)
      3. `Ticker.fast_info.shares` (sometimes set for non-ETFs)
      4. Derived `totalAssets / navPrice` — Yahoo returns None for shares on
         some ETFs (FXF, UUP, CPER, XLC, XLRE observed 2026-06-07) but still
         exposes `totalAssets` and `navPrice` — accurate to within rounding.

    Returns {symbol: shares_outstanding} only for symbols a value was derived for.
    """
    out: dict[str, float] = {}
    for sym in symbols:
        try:
            tkr = yf.Ticker(sym)
            info = tkr.info or {}
            so = info.get("sharesOutstanding") or info.get("shares")
            if not so:
                fi = getattr(tkr, "fast_info", None)
                so = getattr(fi, "shares", None) if fi is not None else None
            if not so:
                # Derive from totalAssets / navPrice. This is the fallback Yahoo
                # leaves available even when sharesOutstanding is None.
                ta = info.get("totalAssets") or info.get("netAssets")
                nav = info.get("navPrice")
                if ta and nav and nav > 0:
                    so = float(ta) / float(nav)
                    log.info("flows: %s shares_out derived from totalAssets/navPrice: %.0f",
                             sym, so)
            if so:
                out[sym] = float(so)
            else:
                log.info("flows: shares_outstanding unavailable for %s (info missing both shares and totalAssets/navPrice)", sym)
        except Exception as exc:
            log.info("flows: shares_outstanding fetch failed for %s: %s", sym, exc)
    return out


def upsert_flows(
    con: duckdb.DuckDBPyConnection,
    rows: list[dict],
) -> int:
    if not rows:
        return 0
    df = pd.DataFrame(rows)
    # Strip diagnostic-only underscore-prefixed keys (e.g. _nav_price) before write.
    diag_cols = [c for c in df.columns if str(c).startswith("_")]
    if diag_cols:
        df = df.drop(columns=diag_cols)
    df["ts"] = pd.to_datetime(df["ts"])
    df["ingested_at"] = datetime.utcnow()
    keep_cols = ["symbol", "ts", "shares_outstanding", "aum_usd", "net_flow_usd",
                 "source", "proxy_method", "confidence", "ingested_at"]
    for c in keep_cols:
        if c not in df.columns:
            df[c] = None
    df = df[keep_cols]
    con.register("incoming_flows", df)
    # Atomic DELETE+INSERT — same rationale as store.upsert_bars: a crash between
    # the auto-committed statements would silently lose the day's flow rows.
    con.execute("BEGIN TRANSACTION")
    try:
        con.execute(
            "DELETE FROM etf_flows WHERE EXISTS "
            "(SELECT 1 FROM incoming_flows i WHERE i.symbol = etf_flows.symbol AND i.ts = etf_flows.ts)"
        )
        con.execute("INSERT INTO etf_flows SELECT * FROM incoming_flows")
        con.execute("COMMIT")
    except Exception:
        con.execute("ROLLBACK")
        raise
    finally:
        con.unregister("incoming_flows")
    return len(df)


def snapshot_today(
    cfg: Config,
    con: duckdb.DuckDBPyConnection,
    asof: date,
) -> dict:
    """Daily ETF-flow snapshot. Strategy:

    - For the 6 PRIORITY_ISSUER_DIRECT ETFs (SPY/QQQ/IWM/GLD/TLT/HYG): use
      `flows_issuer_direct.fetch_issuer_direct` which prefers the issuer's
      own data (SSGA holdings XLSX, fall back to yf.info AUM) — confidence
      0.70-0.80 and `proxy_method` reflects the source.
    - For everything else in the universe: legacy Method B shares-delta proxy
      at confidence 0.6. Universal coverage; the price for using a less
      authoritative source is the lower confidence weight at scoring time.
    """
    etf_classes = {"equity_us", "equity_intl", "equity_sector", "bond", "credit",
                   "commodity_precious", "commodity_industrial", "commodity_energy", "fx"}
    syms = [s.symbol for s in cfg.universe if s.asset_class in etf_classes]
    # HK index ETFs share asset_class=equity_hk with single names (Tencent etc.)
    # in config.yaml; pull them in by explicit symbol whitelist.
    syms.extend(s.symbol for s in cfg.universe if s.symbol in HK_ETF_SYMBOLS)

    close_df = con.execute(
        "SELECT symbol, close FROM raw_bars WHERE ts = ?", [asof]
    ).df()
    close_map = dict(zip(close_df["symbol"], close_df["close"]))

    issuer_rows: list[dict] = []
    issuer_failures: list[str] = []
    for sym in PRIORITY_ISSUER_DIRECT:
        if sym not in syms:  # ignore picks that aren't in our universe
            continue
        row = fetch_issuer_direct(sym, asof, close_map)
        if row is None:
            issuer_failures.append(sym); continue
        issuer_rows.append(row)

    # Reconcile net_flow against the most recent prior AUM per symbol.
    if issuer_rows:
        prior = {}
        for r in issuer_rows:
            p = con.execute(
                "SELECT aum_usd FROM etf_flows "
                "WHERE symbol = ? AND ts < ? AND aum_usd IS NOT NULL "
                "ORDER BY ts DESC LIMIT 1",
                [r["symbol"], r["ts"]],
            ).fetchone()
            if p and p[0]:
                prior[r["symbol"]] = float(p[0])
        issuer_rows = reconcile_net_flow(issuer_rows, prior)

    # Method B for the rest of the universe.
    issuer_done = {r["symbol"] for r in issuer_rows}
    method_b_syms = [s for s in syms if s not in issuer_done]
    snaps = fetch_shares_outstanding(method_b_syms)
    method_b_rows: list[dict] = []
    n_with_delta = 0
    for sym, shares in snaps.items():
        close = close_map.get(sym)
        if close is None or pd.isna(close):
            continue
        aum = float(shares) * float(close)
        prev = con.execute(
            "SELECT shares_outstanding FROM etf_flows "
            "WHERE symbol = ? AND ts < ? AND shares_outstanding IS NOT NULL "
            "ORDER BY ts DESC LIMIT 1",
            [sym, asof],
        ).fetchone()
        net_flow = None
        if prev and prev[0]:
            net_flow = float((shares - float(prev[0])) * close); n_with_delta += 1
        method_b_rows.append({
            "symbol": sym, "ts": asof, "shares_outstanding": shares,
            "aum_usd": aum, "net_flow_usd": net_flow,
            "source": "yfinance", "proxy_method": "shares_delta",
            "confidence": CONFIDENCE_SHARES_DELTA,
        })

    all_rows = issuer_rows + method_b_rows
    n_upserted = upsert_flows(con, all_rows)
    return {
        "asof": asof.isoformat(),
        "n_issuer_direct": len(issuer_rows),
        "n_method_b": len(method_b_rows),
        "n_with_delta": n_with_delta,
        "n_upserted": n_upserted,
        "issuer_failures": issuer_failures,
        "priority_issuer_direct_candidates": list(PRIORITY_ISSUER_DIRECT),
    }


def load_flow_panel(
    con: duckdb.DuckDBPyConnection,
    asof: date,
    lookback_days: int = 90,
) -> pd.DataFrame:
    """Wide panel: index=date, columns=symbol, values=net_flow_usd."""
    start = (pd.Timestamp(asof) - pd.Timedelta(days=lookback_days)).date()
    df = con.execute(
        "SELECT ts, symbol, net_flow_usd FROM etf_flows WHERE ts BETWEEN ? AND ?",
        [start, asof],
    ).df()
    if df.empty:
        return pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"])
    return df.pivot(index="ts", columns="symbol", values="net_flow_usd").sort_index()
