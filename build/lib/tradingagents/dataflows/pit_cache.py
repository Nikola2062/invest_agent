"""Point-in-time keyed disk cache for data-vendor calls.

Rationale: a backtest re-runs the same `(vendor, method, ticker, as_of)`
combination many times across the agent debate, the risk debate, and
reruns of the harness. Without caching, each repeat hits the live API,
costing time and money — and worse, the same call can return different
results on different runs when the live snapshot shifts.

Design:
  - Keyed on ``(vendor, method, ticker_like_arg, as_of, args/kwargs hash)``.
  - **Past-only caching:** when ``as_of`` is today (live mode), the cache
    is bypassed so we always re-fetch. When ``as_of`` is strictly in the
    past, PIT data is immutable, so the cache entry is valid forever.
  - String-valued (matches the existing ``route_to_vendor`` return shape).
  - Disabled by default until config opts in via ``pit_cache_enabled``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Callable

from .config import get_config
from .providers import is_historical
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)


def _cache_root() -> Path | None:
    """Resolve the cache root from config, or None if caching is disabled."""
    config = get_config()
    if not config.get("pit_cache_enabled", False):
        return None
    base = config.get("data_cache_dir")
    if not base:
        return None
    return Path(base) / "pit"


def _key(vendor: str, method: str, args: tuple, kwargs: dict) -> tuple[str, str]:
    """Build a (directory, filename) tuple for the cache entry.

    Ticker is path-sanitised to prevent traversal. The remaining args/kwargs
    are JSON-hashed to keep filenames short and stable.
    """
    ticker = ""
    if args and isinstance(args[0], str):
        ticker = safe_ticker_component(args[0]) or "_"
    payload = json.dumps([args, sorted(kwargs.items())], default=str, sort_keys=True)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]
    return f"{vendor}/{method}/{ticker}", f"{digest}.txt"


def cached_call(
    vendor: str,
    method: str,
    args: tuple,
    kwargs: dict,
    fn: Callable[..., Any],
    as_of: str | None,
) -> Any:
    """Run ``fn(*args, **kwargs)`` with PIT-keyed disk caching.

    Bypasses the cache when caching is disabled, when ``as_of`` is missing,
    or when ``as_of`` is today (live mode — must always re-fetch).
    """
    root = _cache_root()
    if root is None or not as_of or not is_historical(as_of):
        return fn(*args, **kwargs)

    subdir, filename = _key(vendor, method, args, kwargs)
    cache_dir = root / subdir / as_of
    cache_path = cache_dir / filename

    if cache_path.exists():
        try:
            return cache_path.read_text(encoding="utf-8")
        except OSError as e:
            logger.warning("PIT cache read failed for %s: %s — refetching", cache_path, e)

    result = fn(*args, **kwargs)

    # Only persist string results — matches route_to_vendor contract.
    # Non-string results (e.g. dict from a future provider) pass through
    # untouched so we don't accidentally serialise something lossy.
    if isinstance(result, str):
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(result, encoding="utf-8")
        except OSError as e:
            logger.warning("PIT cache write failed for %s: %s", cache_path, e)

    return result
