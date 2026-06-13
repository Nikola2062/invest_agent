"""Shared helpers for agents."""
from __future__ import annotations

from typing import Any


def fmt_optional(v: Any, fmt: str = "{:.2f}", default: str = "n/a") -> str:
    if v is None:
        return default
    try:
        if isinstance(v, float) and (v != v):  # NaN
            return default
        return fmt.format(v)
    except (TypeError, ValueError):
        return default


def fmt_pct(v: Any, default: str = "n/a") -> str:
    if v is None:
        return default
    try:
        if isinstance(v, float) and (v != v):
            return default
        # yfinance returns fractions for margins / growth — auto-detect.
        if isinstance(v, float) and abs(v) < 1.5:
            return f"{v * 100:.1f}%"
        return f"{v:.1f}%"
    except (TypeError, ValueError):
        return default


def fmt_dollars(v: Any, default: str = "n/a") -> str:
    if v is None:
        return default
    try:
        v = float(v)
        if v != v:
            return default
        if abs(v) >= 1e9:
            return f"${v / 1e9:.2f}B"
        if abs(v) >= 1e6:
            return f"${v / 1e6:.2f}M"
        return f"${v:,.0f}"
    except (TypeError, ValueError):
        return default
