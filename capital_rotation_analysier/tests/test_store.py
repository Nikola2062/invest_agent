"""Tests for store.py — the idempotency and replay guarantees are the
load-bearing contract for the whole pipeline."""
from __future__ import annotations

from datetime import date, datetime
from pathlib import Path

import duckdb
import pytest

from rotation.store import connect, upsert_bars


def _row(symbol="SPY", ts=date(2025, 6, 1), close=100.0, **kw) -> dict:
    base = {
        "symbol": symbol, "asset_class": "equity_us", "ts": ts,
        "open": close - 1, "high": close + 1, "low": close - 1.5,
        "close": close, "adj_close": close, "volume": 1_000_000.0,
        "source": "test", "ingested_at": datetime(2025, 6, 1, 22, 0),
        "stale": False,
    }
    base.update(kw)
    return base


def test_upsert_inserts_then_replay_is_noop(tmp_path):
    db = tmp_path / "test.duckdb"
    with connect(db) as con:
        # First insert: 2 rows
        result = upsert_bars(con, [_row(close=100), _row(symbol="QQQ", close=200)])
        assert result.inserted == 2
        assert result.updated == 0

        # Replay identical data: 0 inserts, 0 updates (idempotent)
        result2 = upsert_bars(con, [_row(close=100), _row(symbol="QQQ", close=200)])
        assert result2.inserted == 0, "replay must not re-insert"
        assert result2.updated == 0, "replay must not bump revision"


def test_upsert_bumps_revision_when_data_changes(tmp_path):
    db = tmp_path / "test.duckdb"
    with connect(db) as con:
        upsert_bars(con, [_row(close=100)])
        # Same (symbol, ts) but different close: revision must bump
        result = upsert_bars(con, [_row(close=101)])
        assert result.inserted == 0
        assert result.updated == 1
        row = con.execute("SELECT close, revision FROM raw_bars WHERE symbol='SPY'").fetchone()
        assert row[0] == 101.0
        assert row[1] == 1


def test_upsert_empty_input_noop(tmp_path):
    db = tmp_path / "test.duckdb"
    with connect(db) as con:
        result = upsert_bars(con, [])
        assert result.inserted == 0
        assert result.updated == 0


def test_upsert_handles_mixed_new_and_changed(tmp_path):
    db = tmp_path / "test.duckdb"
    with connect(db) as con:
        # Seed
        upsert_bars(con, [
            _row(symbol="SPY", close=100),
            _row(symbol="QQQ", close=200),
        ])
        # Second batch: SPY changes, QQQ same, IWM new
        r = upsert_bars(con, [
            _row(symbol="SPY", close=101),     # changed -> updated
            _row(symbol="QQQ", close=200),     # same -> noop
            _row(symbol="IWM", close=150),     # new -> inserted
        ])
        assert r.inserted == 1
        assert r.updated == 1
