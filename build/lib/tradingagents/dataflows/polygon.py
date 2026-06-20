"""Polygon.io provider — US prices + ticker-tagged news with real timestamps.

The yfinance equivalents work, but two quality issues motivate Polygon:

  - Splits/dividends: yfinance's auto-adjustment is good but undocumented;
    Polygon makes the adjustment policy explicit (``adjusted=true``).
  - News timestamps: yfinance news ``pub_date`` is unreliable (often
    missing, sometimes wrong by hours). Polygon's ``published_utc`` is
    authoritative — which matters for PIT correctness when injecting
    news into a backtest dated to a specific trading day.

What this module covers:
  - ``get_stock_data``  — daily OHLCV via /v2/aggs (adjusted)
  - ``get_news``        — per-ticker news via /v2/reference/news
  - ``get_global_news`` — broad-query news via the same endpoint

What it does NOT cover (yet):
  - Insider transactions (separate experimental endpoint)
  - Technical indicators (derived from get_stock_data via stockstats)
  - Reference / corporate actions endpoints

Activation:
  1. Sign up at https://polygon.io, set ``POLYGON_API_KEY`` in .env.
  2. In config: ``data_vendors["core_stock_apis"] = "polygon"``
     and/or ``data_vendors["news_data"] = "polygon"``.

Rate limits:
  - Free tier: 5 req/min (effectively unusable for backtests).
  - Stocks Starter ($30/mo): unlimited daily aggregates.
  Hit a 429 and the client retries with backoff; sustained 429s mean the
  caller needs to upgrade their plan, not tune the client.
"""

from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Annotated, Iterable, Optional

import pandas as pd
import requests

from .config import get_config

logger = logging.getLogger(__name__)


_HOST = "https://api.polygon.io"
_DEFAULT_TIMEOUT = 30
_MAX_429_RETRIES = 3


# --- auth + HTTP ---------------------------------------------------------


def _api_key() -> str:
    """Resolve the Polygon API key from env or config. Raises if missing."""
    key = os.environ.get("POLYGON_API_KEY") or get_config().get("polygon_api_key", "")
    if not key:
        raise RuntimeError(
            "Polygon requires an API key. Set POLYGON_API_KEY in your .env "
            "or 'polygon_api_key' in config. Sign up at https://polygon.io."
        )
    return key


def _get_json(path: str, params: Optional[dict] = None) -> dict:
    """Authenticated GET against Polygon. Handles 429 with backoff.

    A 429 means the account's plan limit was hit. We retry with
    exponential backoff up to _MAX_429_RETRIES; persistent 429s indicate
    the caller needs to upgrade their plan (or slow their own loop), so
    we surface the eventual failure rather than silently sleeping forever.
    """
    full_url = f"{_HOST}{path}"
    params = dict(params or {})
    params["apiKey"] = _api_key()

    for attempt in range(_MAX_429_RETRIES + 1):
        resp = requests.get(full_url, params=params, timeout=_DEFAULT_TIMEOUT)
        if resp.status_code == 429:
            if attempt >= _MAX_429_RETRIES:
                raise RuntimeError(
                    f"Polygon 429 after {_MAX_429_RETRIES + 1} attempts on {path}. "
                    "Free tier is 5 req/min — upgrade your plan or reduce request rate."
                )
            delay = 2 ** attempt * 5  # 5s, 10s, 20s
            logger.warning(
                "Polygon 429 on %s (attempt %d/%d) — sleeping %ds",
                path, attempt + 1, _MAX_429_RETRIES + 1, delay,
            )
            time.sleep(delay)
            continue
        resp.raise_for_status()
        return resp.json()
    # Unreachable; loop either returns or raises.
    raise RuntimeError("polygon retry loop exited unexpectedly")


# --- get_stock_data ------------------------------------------------------


def get_stock_data(
    symbol: Annotated[str, "ticker symbol"],
    start_date: Annotated[str, "start date YYYY-MM-DD"],
    end_date: Annotated[str, "end date YYYY-MM-DD"],
) -> str:
    """Daily adjusted OHLCV for ``symbol``, formatted to match the yfinance CSV shape.

    Polygon's aggregates endpoint already returns adjusted prices when
    ``adjusted=true``; no further splits/dividends handling needed. The
    50,000-bar cap is per request — for daily bars this covers ~200
    years, so practical date ranges always fit in one call.
    """
    datetime.strptime(start_date, "%Y-%m-%d")  # validate format
    datetime.strptime(end_date, "%Y-%m-%d")

    symbol = symbol.upper()
    path = f"/v2/aggs/ticker/{symbol}/range/1/day/{start_date}/{end_date}"
    data = _get_json(path, params={"adjusted": "true", "sort": "asc", "limit": 50000})

    results = data.get("results") or []
    if not results:
        return f"No data found for symbol '{symbol}' between {start_date} and {end_date}"

    # Polygon returns t as Unix ms; convert to date strings.
    rows = []
    for bar in results:
        ts = pd.to_datetime(bar["t"], unit="ms")
        rows.append({
            "Date": ts.strftime("%Y-%m-%d"),
            "Open": round(float(bar.get("o", 0)), 2),
            "High": round(float(bar.get("h", 0)), 2),
            "Low": round(float(bar.get("l", 0)), 2),
            "Close": round(float(bar.get("c", 0)), 2),
            "Volume": int(bar.get("v", 0)),
            "VWAP": round(float(bar.get("vw", 0)), 2) if "vw" in bar else None,
        })

    df = pd.DataFrame(rows).set_index("Date")
    header = (
        f"# Stock data for {symbol} from {start_date} to {end_date}\n"
        f"# Total records: {len(df)}\n"
        f"# Source: Polygon.io (adjusted=true)\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + df.to_csv()


# --- news endpoints ------------------------------------------------------


def _news_request(params: dict, limit: int) -> list[dict]:
    """Page through /v2/reference/news until ``limit`` is reached or no next page."""
    out: list[dict] = []
    next_url = None
    per_page = min(limit, 1000)  # Polygon's per-page max is 1000.
    request_params = {**params, "limit": per_page}

    while len(out) < limit:
        if next_url:
            # Polygon's next_url already includes its own query string; just
            # append the API key.
            resp = requests.get(
                next_url,
                params={"apiKey": _api_key()},
                timeout=_DEFAULT_TIMEOUT,
            )
            if resp.status_code == 429:
                logger.warning("Polygon 429 on news pagination — stopping early")
                break
            resp.raise_for_status()
            data = resp.json()
        else:
            data = _get_json("/v2/reference/news", params=request_params)

        page = data.get("results") or []
        if not page:
            break
        out.extend(page)
        next_url = data.get("next_url")
        if not next_url:
            break

    return out[:limit]


def _format_articles(articles: Iterable[dict]) -> str:
    """Render Polygon news rows to the markdown shape the yfinance formatter emits."""
    parts = []
    for art in articles:
        title = art.get("title", "No title")
        publisher = (art.get("publisher") or {}).get("name", "Unknown")
        description = art.get("description", "")
        url = art.get("article_url", "")
        parts.append(f"### {title} (source: {publisher})")
        if description:
            parts.append(description)
        if url:
            parts.append(f"Link: {url}")
        parts.append("")
    return "\n".join(parts).rstrip() + "\n"


def get_news(
    ticker: Annotated[str, "ticker symbol"],
    start_date: Annotated[str, "start date YYYY-MM-DD"],
    end_date: Annotated[str, "end date YYYY-MM-DD"],
) -> str:
    """Ticker-tagged news from Polygon, filtered to the requested window.

    Filters server-side on ``published_utc`` so we don't pull the entire
    article history. The endpoint sorts descending by ``published_utc``;
    we re-render to match the existing markdown shape.
    """
    config = get_config()
    article_limit = config.get("news_article_limit", 20)

    ticker = ticker.upper()
    # Polygon expects ISO-8601 (date-only is accepted) with .gte / .lte suffixes.
    params = {
        "ticker": ticker,
        "published_utc.gte": start_date,
        "published_utc.lte": end_date,
        "order": "desc",
        "sort": "published_utc",
    }
    articles = _news_request(params, article_limit)
    if not articles:
        return f"No news found for {ticker} between {start_date} and {end_date}"
    body = _format_articles(articles)
    return f"## {ticker} News, from {start_date} to {end_date}:\n\n{body}"


def get_global_news(
    curr_date: Annotated[str, "current date YYYY-MM-DD"],
    look_back_days: Annotated[Optional[int], "days to look back"] = None,
    limit: Annotated[Optional[int], "max articles"] = None,
) -> str:
    """Broad macro news via Polygon search over the configured queries.

    Polygon's news endpoint doesn't have a full-text search parameter,
    so we approximate global news by pulling untagged recent articles
    within the lookback window. The result is sorted descending by
    ``published_utc`` and trimmed to ``limit``.
    """
    config = get_config()
    if look_back_days is None:
        look_back_days = config.get("global_news_lookback_days", 7)
    if limit is None:
        limit = config.get("global_news_article_limit", 10)

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start = (curr_dt - pd.Timedelta(days=look_back_days)).strftime("%Y-%m-%d")

    params = {
        "published_utc.gte": start,
        "published_utc.lte": curr_date,
        "order": "desc",
        "sort": "published_utc",
    }
    articles = _news_request(params, limit)
    if not articles:
        return f"No global news found for {curr_date}"
    body = _format_articles(articles)
    return f"## Global Market News, from {start} to {curr_date}:\n\n{body}"
