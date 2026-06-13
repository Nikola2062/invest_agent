from __future__ import annotations

from datetime import datetime, date
from typing import Iterable

import pandas as pd


def normalize_yfinance_frame(
    df: pd.DataFrame,
    symbol: str,
    asset_class: str,
    source: str = "yfinance",
    ingested_at: datetime | None = None,
) -> list[dict]:
    """yfinance returns a DataFrame with columns Open/High/Low/Close/Adj Close/Volume.

    With multi-symbol downloads it can be a column-MultiIndex; the caller is
    expected to slice down to a single-symbol frame before calling this.
    """
    if df is None or df.empty:
        return []

    ingested_at = ingested_at or datetime.utcnow()
    rows: list[dict] = []

    # Some yfinance returns drop "Adj Close" depending on version. Fall back to Close.
    has_adj = "Adj Close" in df.columns

    for ts, r in df.iterrows():
        ts_date = ts.date() if hasattr(ts, "date") else ts
        if pd.isna(r.get("Close")):
            continue
        rows.append({
            "symbol": symbol,
            "asset_class": asset_class,
            "ts": ts_date,
            "open": _f(r.get("Open")),
            "high": _f(r.get("High")),
            "low": _f(r.get("Low")),
            "close": _f(r.get("Close")),
            "adj_close": _f(r.get("Adj Close")) if has_adj else _f(r.get("Close")),
            "volume": _f(r.get("Volume")),
            "source": source,
            "ingested_at": ingested_at,
            "stale": False,
        })
    return rows


def _f(v) -> float | None:
    if v is None or pd.isna(v):
        return None
    return float(v)
