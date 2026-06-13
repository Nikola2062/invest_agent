from __future__ import annotations

from datetime import date, timedelta
from typing import Iterable


class ValidationIssue(Exception):
    pass


def validate_row(row: dict, outlier_intraday_pct: float) -> tuple[bool, str]:
    """Returns (is_valid, reason_if_not). Reason is empty on success.

    Checks (cheap, single-row):
    - required OHLCV present and non-negative
    - high >= max(open, close, low); low <= min(open, close, high)
    - intraday range <= outlier_intraday_pct% of close (catches split/dividend not yet applied)
    """
    for col in ("open", "high", "low", "close"):
        v = row.get(col)
        if v is None:
            return False, f"missing_{col}"
        if v < 0:
            return False, f"negative_{col}"

    o, h, l, c = row["open"], row["high"], row["low"], row["close"]
    if h < max(o, c, l) - 1e-9:
        return False, "high_below_oc_or_low"
    if l > min(o, c, h) + 1e-9:
        return False, "low_above_oc_or_high"

    if c > 0:
        intraday_pct = 100.0 * (h - l) / c
        if intraday_pct > outlier_intraday_pct:
            return False, f"intraday_range_{intraday_pct:.1f}pct"

    vol = row.get("volume")
    if vol is not None and vol < 0:
        return False, "negative_volume"

    return True, ""


def is_stale(row_ts: date, asof: date, threshold_days: int) -> bool:
    return (asof - row_ts).days > threshold_days
