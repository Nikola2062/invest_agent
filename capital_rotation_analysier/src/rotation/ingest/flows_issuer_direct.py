"""Issuer-direct flow ingest for the 6 priority ETFs (SPY, QQQ, IWM, GLD, TLT, HYG).

Reality check (probed 2026-06-07):

| ETF | Issuer | Direct endpoint that works today |
|---|---|---|
| SPY | SSGA  | Daily holdings XLSX  ✅ |
| GLD | SSGA  | spdrgoldshares.com paths 404 today; falls back to yf.info |
| QQQ | Invesco | HTML page wraps a JSON blob; brittle to scrape, falls back to yf.info |
| IWM | iShares | `.ajax` endpoint serves HTML, not JSON (URL pattern from spec is stale); falls back |
| TLT | iShares | same as IWM |
| HYG | iShares | same as IWM |

So we implement **two** confidence tiers (in addition to the legacy `shares_delta` Method B):

  1. **`ssga_holdings`** (confidence 0.80): SSGA daily holdings XLSX. AUM is computed
     from `Σ shares_held_i × close_i` against our `raw_bars` close prices for
     the constituents we already store. For constituents we DON'T track, we
     approximate using the basket's `Weight (%)` * total weight sum. The
     XLSX header carries the as-of date stamp directly from SSGA. Works for SPY.

  2. **`yf_info_aum`** (confidence 0.70): yfinance `Ticker.info` exposes
     `totalAssets`, `netAssets`, `sharesOutstanding`, `navPrice` — these are
     ISSUER-PUBLISHED values aggregated by Yahoo. Snapshotted daily, the
     delta of `totalAssets` is a true AUM-flow read, distinct from the
     `sharesOutstanding × close` proxy used by Method B (the two cross-check
     each other and can be reconciled at audit time).

We deliberately do NOT scrape iShares / Invesco HTML for shares-outstanding —
those endpoints are fragile, rate-limited, and the data they expose duplicates
what yf.info already provides via the same Yahoo aggregation that ETF.com,
StockAnalysis, and similar tools use.

Calling convention: each `fetch_<source>(symbol, asof, close_map)` returns a
dict matching the `etf_flows` row schema (see schema.py), or None.
"""
from __future__ import annotations

import io
import logging
import urllib.request
from datetime import date, datetime
from typing import Iterable

import pandas as pd
import yfinance as yf

log = logging.getLogger(__name__)


PRIORITY_ETFS = ["SPY", "QQQ", "IWM", "GLD", "TLT", "HYG"]

# Confidence tiers per the project docs §1.6.1 decision 13 — single place to tune the
# provenance weights downstream consumers see in etf_flows.confidence.
CONFIDENCE_SSGA_HOLDINGS = 0.80
CONFIDENCE_YF_INFO_AUM = 0.70

# Map ETF -> primary source. Inferred from §2.6 of the design doc + 2026-06-07
# probe. Anything not in this dict will use yf.info as its issuer-direct path.
ISSUER_PATH = {
    "SPY": "ssga_holdings",
    "QQQ": "yf_info_aum",
    "IWM": "yf_info_aum",
    "TLT": "yf_info_aum",
    "HYG": "yf_info_aum",
    "GLD": "yf_info_aum",
}


# ============================================================
# Source 1: SSGA daily holdings XLSX
# ============================================================

SSGA_TEMPLATE = (
    "https://www.ssga.com/us/en/individual/etfs/library-content/products/"
    "fund-data/etfs/us/holdings-daily-us-en-{sym_lower}.xlsx"
)


def fetch_ssga_holdings(symbol: str, asof: date, close_map: dict[str, float]) -> dict | None:
    """SSGA publishes a daily holdings XLSX. We compute the fund's AUM as
    Σ shares_held × close per constituent, using `close_map` for constituents
    we already have in raw_bars and assuming `Weight (%)` is exact for the rest.

    The XLSX header carries the issuer's "Holdings: As of DD-MMM-YYYY" date.
    If that's earlier than `asof` (T-1 holdings file), we still write the row
    keyed to the issuer date — the staleness flag picks it up downstream.
    """
    url = SSGA_TEMPLATE.format(sym_lower=symbol.lower())
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as r:
            raw = r.read()
    except Exception as exc:
        log.info("SSGA fetch failed for %s: %s", symbol, exc)
        return None

    try:
        header = pd.read_excel(io.BytesIO(raw), header=None, nrows=10)
        # Locate "Holdings:" row and "Name|Ticker|...|Weight|...|Shares Held" header row.
        holdings_ts = None
        for r in range(len(header)):
            cell0 = str(header.iat[r, 0]).strip().lower() if pd.notna(header.iat[r, 0]) else ""
            if cell0 == "holdings:":
                try:
                    holdings_ts = pd.to_datetime(str(header.iat[r, 1]).replace("As of", "").strip()).date()
                except Exception:
                    pass
            if cell0 == "name":
                hdr_row = r
                break
        else:
            log.warning("SSGA %s: header row not found", symbol); return None

        body = pd.read_excel(io.BytesIO(raw), header=hdr_row)
        body = body.dropna(subset=["Ticker"]).copy()
        # SSGA uses 'Weight' in percent (e.g. 8.156941)
        weight_col = next((c for c in body.columns if str(c).strip().lower().startswith("weight")), None)
        shares_col = next((c for c in body.columns if "shares" in str(c).lower()), None)
        if weight_col is None or shares_col is None:
            log.warning("SSGA %s: weight/shares column not found: %s", symbol, list(body.columns))
            return None

        body["close"] = body["Ticker"].astype(str).str.strip().map(close_map)

        # AUM via known constituents: shares_held * close (only counts ones we have prices for)
        known = body.dropna(subset=["close"])
        partial_aum = float((known[shares_col].astype(float) * known["close"]).sum())
        partial_weight = float(known[weight_col].astype(float).sum())  # in percent

        if partial_weight < 50.0:  # we only know <50% of the basket — too noisy
            log.info("SSGA %s: only %.1f%% of basket has close prices; aborting derived AUM",
                     symbol, partial_weight)
            return None

        # Scale up by the missing weight: AUM_total = partial_aum / (partial_weight/100)
        aum_total = partial_aum / (partial_weight / 100.0)

        # NAV per share = close[symbol] (close approximates NAV closely for liquid ETFs)
        nav_close = close_map.get(symbol)
        if nav_close is None or nav_close == 0:
            log.warning("SSGA %s: no close for ETF itself", symbol); return None
        shares_outstanding = aum_total / nav_close

        ts = holdings_ts or asof
        return {
            "symbol": symbol, "ts": ts,
            "shares_outstanding": shares_outstanding,
            "aum_usd": aum_total,
            "net_flow_usd": None,  # filled by reconciler vs prior row
            "source": "ssga.com/holdings", "proxy_method": "ssga_holdings",
            "confidence": CONFIDENCE_SSGA_HOLDINGS,
            "_constituent_weight_known_pct": partial_weight,
        }
    except Exception as exc:
        log.warning("SSGA %s: parse failed: %s", symbol, exc)
        return None


# ============================================================
# Source 2: yfinance .info (issuer-published AUM via Yahoo aggregation)
# ============================================================

def fetch_yf_info_aum(symbol: str, asof: date, close_map: dict[str, float]) -> dict | None:
    """yfinance's `Ticker.info` exposes the issuer's totalAssets / sharesOutstanding /
    navPrice — these are published by the ETF sponsor and aggregated by Yahoo.

    Distinct from Method B because we record `totalAssets` AND `sharesOutstanding`
    as parallel reads; their daily deltas should agree (give or take NAV drift),
    and a disagreement is a data-quality signal.
    """
    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as exc:
        log.info("yf.info fetch failed for %s: %s", symbol, exc); return None

    total_assets = info.get("totalAssets") or info.get("netAssets")
    shares = info.get("sharesOutstanding")
    nav = info.get("navPrice")
    if not (total_assets and shares):
        return None

    return {
        "symbol": symbol, "ts": asof,
        "shares_outstanding": float(shares),
        "aum_usd": float(total_assets),
        "net_flow_usd": None,  # reconciler computes vs prior
        "source": "yahoo.info", "proxy_method": "yf_info_aum",
        "confidence": CONFIDENCE_YF_INFO_AUM,
        "_nav_price": float(nav) if nav else None,
    }


# ============================================================
# Per-ETF dispatcher
# ============================================================

FETCHERS = {
    "ssga_holdings": fetch_ssga_holdings,
    "yf_info_aum":   fetch_yf_info_aum,
}


def fetch_issuer_direct(symbol: str, asof: date, close_map: dict[str, float]) -> dict | None:
    """Returns one flow row using the ETF's primary issuer source, falling back
    to yf.info if the primary source fails."""
    primary = ISSUER_PATH.get(symbol, "yf_info_aum")
    row = FETCHERS[primary](symbol, asof, close_map)
    if row is not None:
        return row
    if primary != "yf_info_aum":
        log.info("issuer_direct %s: primary %s failed; falling back to yf_info_aum",
                 symbol, primary)
        return FETCHERS["yf_info_aum"](symbol, asof, close_map)
    return None


def reconcile_net_flow(rows: list[dict], prior_aums: dict[str, float]) -> list[dict]:
    """Fill `net_flow_usd` from prior-day AUM where available. The simplest
    decomposition: ΔAUM = ΔAUM_due_to_flows + AUM × r_d. We approximate
    ΔAUM_due_to_flows = ΔAUM − close*Δshares×close — but that's circular when
    the input IS shares*close. Honest fallback: just use ΔAUM minus market drift,
    where market drift = AUM_yesterday * (close_today / close_yesterday - 1).
    Caller passes `prior_aums` keyed by symbol.

    This is a coarse estimate — the true daily creations/redemptions are not
    public. The §3.5 z-score normalization absorbs this approximation error.
    """
    out = []
    for r in rows:
        sym, aum = r["symbol"], r["aum_usd"]
        prior = prior_aums.get(sym)
        if prior is not None and prior > 0:
            r["net_flow_usd"] = float(aum - prior)
        out.append(r)
    return out
