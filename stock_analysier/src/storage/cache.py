"""Persistent SQLite TTL cache for upstream fetches.

The Finnhub free tier rate-limits at 60 calls/min and yfinance's `.info` is slow
and occasionally flaky. With 4 daily Telegram digests × N watchlist tickers,
even a modest watchlist burns the quota by lunch. This cache survives across
pipeline runs so the deterministic part of the system isn't being stymied by
upstream backpressure.

Storage: `data/cache.sqlite` (gitignored). One table, pickle-encoded values.
Single-user, single-process — no locking concerns.
"""
from __future__ import annotations

import logging
import pickle
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, TypeVar

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE_PATH = PROJECT_ROOT / "data" / "cache.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache (
    key TEXT PRIMARY KEY,
    value_pickle BLOB NOT NULL,
    expires_at_utc TEXT NOT NULL,
    stored_at_utc TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_expires ON cache(expires_at_utc);
"""

T = TypeVar("T")


@contextmanager
def _conn(db_path: Path = DEFAULT_CACHE_PATH) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db_path, isolation_level=None)
    c.row_factory = sqlite3.Row
    try:
        c.executescript(_SCHEMA)
        yield c
    finally:
        c.close()


def get(key: str, db_path: Path = DEFAULT_CACHE_PATH) -> Optional[Any]:
    """Return cached value if present and unexpired, else None."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn(db_path) as c:
        row = c.execute(
            "SELECT value_pickle FROM cache WHERE key = ? AND expires_at_utc > ?",
            (key, now),
        ).fetchone()
    if row is None:
        return None
    try:
        return pickle.loads(row["value_pickle"])
    except Exception as e:
        log.warning("Cache decode failed for %s: %s — treating as miss", key, e)
        return None


def put(key: str, value: Any, ttl_seconds: int, db_path: Path = DEFAULT_CACHE_PATH) -> None:
    """Store value under key with an expiry of now + ttl_seconds."""
    now = datetime.now(timezone.utc)
    expires = (now + timedelta(seconds=ttl_seconds)).isoformat()
    blob = pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL)
    with _conn(db_path) as c:
        c.execute(
            "INSERT OR REPLACE INTO cache (key, value_pickle, expires_at_utc, stored_at_utc) "
            "VALUES (?, ?, ?, ?)",
            (key, blob, expires, now.isoformat()),
        )


def invalidate(key: str, db_path: Path = DEFAULT_CACHE_PATH) -> None:
    with _conn(db_path) as c:
        c.execute("DELETE FROM cache WHERE key = ?", (key,))


def purge_expired(db_path: Path = DEFAULT_CACHE_PATH) -> int:
    """Remove expired rows. Returns count deleted. Safe to call anytime."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn(db_path) as c:
        cur = c.execute("DELETE FROM cache WHERE expires_at_utc <= ?", (now,))
        return cur.rowcount or 0


def cached_call(
    key: str, ttl_seconds: int, fn: Callable[[], T], db_path: Path = DEFAULT_CACHE_PATH
) -> T:
    """Helper: return cached value if fresh, otherwise call `fn`, cache, return.

    `fn` is called only on cache miss. Any exception from `fn` is re-raised
    (and nothing is cached). The caller is responsible for choosing a stable key
    that encodes all inputs that affect the result.
    """
    hit = get(key, db_path=db_path)
    if hit is not None:
        log.debug("cache HIT: %s", key)
        return hit  # type: ignore[no-any-return]
    log.debug("cache MISS: %s", key)
    value = fn()
    if value is not None:
        put(key, value, ttl_seconds=ttl_seconds, db_path=db_path)
    return value
