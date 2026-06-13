from __future__ import annotations

import logging
import time
from datetime import date, datetime, timedelta
from typing import Iterable

import pandas as pd
import yfinance as yf

from ..config import Config
from .normalizer import normalize_yfinance_frame

log = logging.getLogger(__name__)


def fetch_bars(
    cfg: Config,
    asof: date,
    symbols: Iterable[str] | None = None,
    lookback_days: int = 5,
) -> list[dict]:
    """Fetch bars from yfinance for the given asof date.

    Pulls a small lookback window so we tolerate weekend/holiday and so the
    adjusted-close revisions on T+1 are picked up on subsequent runs.
    Filters to ts == asof at the end.

    Returns a flat list of normalized bar dicts ready for upsert_bars().
    """
    symbols = list(symbols) if symbols is not None else cfg.symbols()
    if not symbols:
        return []

    start = asof - timedelta(days=lookback_days + 5)  # buffer for non-trading days
    end = asof + timedelta(days=1)                    # yfinance end is exclusive

    asset_class_of = {s: cfg.asset_class(s) for s in symbols}

    rows: list[dict] = []
    ingested_at = datetime.utcnow()

    for attempt in range(cfg.ingest.retry.max_attempts):
        try:
            df = yf.download(
                tickers=" ".join(symbols),
                start=start.isoformat(),
                end=end.isoformat(),
                interval="1d",
                group_by="ticker",
                auto_adjust=False,
                progress=False,
                threads=True,
            )
            break
        except Exception as exc:  # network, parse, rate-limit
            wait = cfg.ingest.retry.backoff_seconds[
                min(attempt, len(cfg.ingest.retry.backoff_seconds) - 1)
            ]
            log.warning("yfinance fetch attempt %d failed: %s; sleeping %ds", attempt + 1, exc, wait)
            if attempt == cfg.ingest.retry.max_attempts - 1:
                raise
            time.sleep(wait)
    else:
        return []

    if df is None or df.empty:
        return []

    # yfinance >=0.2.30 always returns a 2-level column index when group_by='ticker':
    # (ticker, field) for multi-ticker, and a single-ticker download is still a
    # MultiIndex (Price, Ticker). Use xs() to slice down to a flat frame per symbol.
    if isinstance(df.columns, pd.MultiIndex):
        tickers_in_cols = set(df.columns.get_level_values(0).unique()) | set(
            df.columns.get_level_values(1).unique()
        )
    else:
        tickers_in_cols = set(symbols)

    for sym in symbols:
        try:
            if isinstance(df.columns, pd.MultiIndex):
                if sym in df.columns.get_level_values(0):
                    sub = df[sym]
                elif sym in df.columns.get_level_values(1):
                    sub = df.xs(sym, axis=1, level=1)
                else:
                    log.warning("yfinance: no data returned for %s", sym)
                    continue
            else:
                sub = df
            sub = sub.dropna(how="all")
            rows.extend(
                normalize_yfinance_frame(
                    sub, sym, asset_class_of[sym], ingested_at=ingested_at
                )
            )
        except Exception as exc:
            log.warning("yfinance: normalize failed for %s: %s", sym, exc)
            continue

    # Keep the full lookback in the DB — useful for replay diagnostics — but
    # callers that want only the asof row can filter on ts.
    return rows
