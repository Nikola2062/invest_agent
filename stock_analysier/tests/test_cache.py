"""Tests for src/storage/cache.py — the persistent SQLite TTL cache.

The cache wraps fetch_fundamentals + fetch_catalysts so the 4 daily Telegram
digests share one upstream fetch per ticker, not four.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage import cache as kv_cache


@pytest.fixture
def cache_path(tmp_path_factory):
    return tmp_path_factory.mktemp("cache") / "cache.sqlite"


def test_get_returns_none_on_miss(cache_path):
    assert kv_cache.get("never-set", db_path=cache_path) is None


def test_put_then_get_roundtrip(cache_path):
    value = {"foo": "bar", "list": [1, 2, 3], "nested": {"k": 7.5}}
    kv_cache.put("k1", value, ttl_seconds=60, db_path=cache_path)
    assert kv_cache.get("k1", db_path=cache_path) == value


def test_expired_value_returns_none(cache_path):
    kv_cache.put("k2", "stale", ttl_seconds=0, db_path=cache_path)
    # Tiny sleep to ensure the stored expiry is in the past
    time.sleep(0.01)
    assert kv_cache.get("k2", db_path=cache_path) is None


def test_invalidate_removes_value(cache_path):
    kv_cache.put("k3", 42, ttl_seconds=600, db_path=cache_path)
    assert kv_cache.get("k3", db_path=cache_path) == 42
    kv_cache.invalidate("k3", db_path=cache_path)
    assert kv_cache.get("k3", db_path=cache_path) is None


def test_purge_expired_counts(cache_path):
    kv_cache.put("alive", "x", ttl_seconds=600, db_path=cache_path)
    kv_cache.put("dead1", "y", ttl_seconds=0, db_path=cache_path)
    kv_cache.put("dead2", "z", ttl_seconds=0, db_path=cache_path)
    time.sleep(0.01)
    n = kv_cache.purge_expired(db_path=cache_path)
    assert n == 2
    assert kv_cache.get("alive", db_path=cache_path) == "x"


def test_cached_call_only_invokes_fn_once_on_hit(cache_path):
    """Hitting the cache must skip the function entirely — the whole point of caching."""
    calls = {"n": 0}

    def expensive():
        calls["n"] += 1
        return {"answer": 42}

    a = kv_cache.cached_call("key", ttl_seconds=60, fn=expensive, db_path=cache_path)
    b = kv_cache.cached_call("key", ttl_seconds=60, fn=expensive, db_path=cache_path)

    assert a == b == {"answer": 42}
    assert calls["n"] == 1, f"fn invoked {calls['n']} times — cache miss after set"


def test_cached_call_recomputes_after_expiry(cache_path):
    calls = {"n": 0}

    def expensive():
        calls["n"] += 1
        return calls["n"]

    a = kv_cache.cached_call("key", ttl_seconds=0, fn=expensive, db_path=cache_path)
    time.sleep(0.01)
    b = kv_cache.cached_call("key", ttl_seconds=0, fn=expensive, db_path=cache_path)

    assert a == 1
    assert b == 2
    assert calls["n"] == 2


def test_cached_call_does_not_cache_none(cache_path):
    """If fn returns None, treat it as a transient failure — don't cache it."""
    calls = {"n": 0}

    def returns_none():
        calls["n"] += 1
        return None

    kv_cache.cached_call("none-key", ttl_seconds=60, fn=returns_none, db_path=cache_path)
    kv_cache.cached_call("none-key", ttl_seconds=60, fn=returns_none, db_path=cache_path)
    assert calls["n"] == 2


def test_cached_call_propagates_exceptions(cache_path):
    """Failed upstream fetches must not silently cache a sentinel — they must raise."""
    def boom():
        raise RuntimeError("upstream down")

    with pytest.raises(RuntimeError, match="upstream down"):
        kv_cache.cached_call("boom-key", ttl_seconds=60, fn=boom, db_path=cache_path)
    # And the failure must not have created an entry
    assert kv_cache.get("boom-key", db_path=cache_path) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
