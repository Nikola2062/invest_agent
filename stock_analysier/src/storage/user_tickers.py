"""User-added tickers — persistent free-form watchlist outside of config/universe.yaml.

Stored in data/user_tickers.json so it survives across runs without polluting git
(data/ is gitignored). Schema:

    {"US": ["TSLA", "AAPL"], "HK": ["0939.HK"]}
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
STORE_PATH = PROJECT_ROOT / "data" / "user_tickers.json"

Market = Literal["US", "HK"]
_VALID_MARKETS: tuple[str, ...] = ("US", "HK")


def detect_market(symbol: str) -> Market:
    """Infer market from symbol shape — .HK suffix or pure-digit code → HK, else US."""
    s = symbol.strip().upper()
    if s.endswith(".HK"):
        return "HK"
    if re.fullmatch(r"\d{1,5}", s):
        return "HK"
    return "US"


def normalize_symbol(symbol: str, market: Market) -> str:
    """Canonicalize a user-typed symbol. HK numeric codes get zero-padded + .HK suffix."""
    s = symbol.strip().upper()
    if market == "HK":
        if s.endswith(".HK"):
            base = s[:-3]
        else:
            base = s
        if re.fullmatch(r"\d{1,5}", base):
            base = base.zfill(4)
        return f"{base}.HK"
    return s


def _read_raw() -> dict[str, list[str]]:
    if not STORE_PATH.exists():
        return {"US": [], "HK": []}
    try:
        with open(STORE_PATH) as f:
            data = json.load(f) or {}
    except (json.JSONDecodeError, OSError):
        return {"US": [], "HK": []}
    return {m: list(data.get(m, [])) for m in _VALID_MARKETS}


def _write_raw(data: dict[str, list[str]]) -> None:
    STORE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(STORE_PATH, "w") as f:
        json.dump({m: list(data.get(m, [])) for m in _VALID_MARKETS}, f, indent=2)


def load_user_tickers() -> dict[str, list[str]]:
    """Return {"US": [...], "HK": [...]} of user-added symbols (canonicalized)."""
    return _read_raw()


def add_user_ticker(symbol: str, market: Market) -> str:
    """Add a symbol to the persistent user list. Returns the canonicalized form.

    Raises ValueError on bad input or if the symbol already exists in the user list
    for the given market (case-insensitive).
    """
    if market not in _VALID_MARKETS:
        raise ValueError(f"Unknown market {market!r}. Use one of {_VALID_MARKETS}.")
    if not symbol or not symbol.strip():
        raise ValueError("Symbol cannot be empty.")
    canonical = normalize_symbol(symbol, market)
    data = _read_raw()
    if canonical in data[market]:
        raise ValueError(f"{canonical} is already in your {market} list.")
    data[market].append(canonical)
    _write_raw(data)
    return canonical


def remove_user_ticker(symbol: str, market: Market) -> bool:
    """Remove a symbol from the user list. Returns True if removed, False if not found."""
    if market not in _VALID_MARKETS:
        return False
    data = _read_raw()
    canonical = normalize_symbol(symbol, market)
    if canonical not in data[market]:
        return False
    data[market].remove(canonical)
    _write_raw(data)
    return True
