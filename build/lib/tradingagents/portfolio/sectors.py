"""Ticker → sector mapping, fetched once and cached to disk.

Sector data is needed by ``risk_aware_sizer``'s per-sector cap. yfinance
exposes a ``sector`` field on ``Ticker.info`` but that lookup is slow and
rate-limited, so we cache the result so a long backtest doesn't refetch
on every rebalance.

PIT caveat: yfinance reports the *current* sector, not the historical
one. Reclassifications happen rarely (GICS reshuffles every few years)
so the impact on a 3-year backtest is small, but it's a known compromise
documented for the user.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from tradingagents.dataflows.config import get_config
from tradingagents.dataflows.stockstats_utils import yf_retry

logger = logging.getLogger(__name__)


def _cache_path() -> Path:
    base = get_config().get("data_cache_dir", ".")
    return Path(base) / "sectors.json"


def _load_cache() -> dict[str, str]:
    p = _cache_path()
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Could not read sector cache at %s: %s", p, exc)
        return {}


def _save_cache(mapping: dict[str, str]) -> None:
    p = _cache_path()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write sector cache to %s: %s", p, exc)


def fetch_sectors(tickers: Iterable[str], *, refresh: bool = False) -> dict[str, str]:
    """Return a ``{ticker: sector}`` map, fetching uncached tickers from yfinance.

    Set ``refresh=True`` to force a re-fetch (e.g. after a known
    reclassification). Tickers whose lookup fails or returns nothing are
    recorded as ``"Unknown"`` so subsequent calls don't keep retrying
    them; delete ``sectors.json`` to undo this.
    """
    import yfinance as yf

    cache = {} if refresh else _load_cache()
    tickers = list(dict.fromkeys(tickers))
    missing = [t for t in tickers if t not in cache]

    if not missing:
        return {t: cache[t] for t in tickers}

    logger.info("Fetching sector info for %d new ticker(s)", len(missing))
    for ticker in missing:
        try:
            info = yf_retry(lambda t=ticker: yf.Ticker(t).info)
            cache[ticker] = (info or {}).get("sector") or "Unknown"
        except Exception as exc:
            logger.warning("Could not resolve sector for %s: %s", ticker, exc)
            cache[ticker] = "Unknown"

    _save_cache(cache)
    return {t: cache[t] for t in tickers}
