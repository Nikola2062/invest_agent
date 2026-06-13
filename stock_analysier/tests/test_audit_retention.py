"""Tests for src.storage.audit.prune_old_runs.

The audit DB grows ~20 rows/day with the default 4 daily pushes; without a
retention policy the SQLite file grows unboundedly. prune_old_runs deletes
old rows while keeping a per-symbol floor so cold-start guards still function.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.storage import audit


@pytest.fixture
def db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("audit_prune") / "audit.sqlite"


def _seed_run(symbol: str, days_ago: int, db_path: Path, level: int = 1) -> int:
    """Insert a synthetic audit row at a specific age. Returns the row id."""
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    with audit._conn(db_path) as c:
        cur = c.execute(
            """INSERT INTO analysis_runs
               (symbol, market, timestamp_utc, current_price, currency,
                tactical_level, full_result_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (symbol, "US", ts, 50.0, "USD", level, "{}"),
        )
        return int(cur.lastrowid)


def _seed_alert(symbol: str, days_ago: int, db_path: Path) -> int:
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    with audit._conn(db_path) as c:
        cur = c.execute(
            """INSERT INTO thesis_break_alerts
               (symbol, detected_at_utc, severity, category, summary, evidence_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (symbol, ts, "info", "test", "test alert", "{}"),
        )
        return int(cur.lastrowid)


def _count_runs(db_path: Path) -> int:
    with audit._conn(db_path) as c:
        return int(c.execute("SELECT COUNT(*) FROM analysis_runs").fetchone()[0])


def _count_alerts(db_path: Path) -> int:
    with audit._conn(db_path) as c:
        return int(c.execute("SELECT COUNT(*) FROM thesis_break_alerts").fetchone()[0])


def test_prune_removes_old_rows_outside_floor(db_path):
    """Rows older than cutoff AND past the per-symbol floor are deleted."""
    # Symbol A: 50 runs spanning 400d ago to 1d ago
    for i in range(50):
        _seed_run("A", days_ago=i * 8, db_path=db_path)
    assert _count_runs(db_path) == 50

    result = audit.prune_old_runs(days_to_keep=180, keep_min_per_symbol=10, db_path=db_path)

    # Should keep: any row newer than 180d + the most-recent 10 per symbol
    remaining = _count_runs(db_path)
    assert remaining < 50
    assert remaining >= 10, "must keep at least keep_min_per_symbol rows"
    assert result["runs_deleted"] == 50 - remaining


def test_prune_keeps_recent_rows(db_path):
    """Rows newer than cutoff are NEVER deleted."""
    for i in range(20):
        _seed_run("B", days_ago=i, db_path=db_path)  # 0..19 days old
    audit.prune_old_runs(days_to_keep=30, keep_min_per_symbol=5, db_path=db_path)
    assert _count_runs(db_path) == 20  # all younger than 30d


def test_prune_respects_per_symbol_floor(db_path):
    """Even very old rows are kept if they're in the per-symbol most-recent N set."""
    # Symbol C: only 5 runs, all very old (400d+)
    for i in range(5):
        _seed_run("C", days_ago=400 + i, db_path=db_path)

    audit.prune_old_runs(days_to_keep=30, keep_min_per_symbol=10, db_path=db_path)

    # Floor is 10 but only 5 exist — all 5 must survive even though they're old
    assert _count_runs(db_path) == 5


def test_prune_deletes_old_alerts(db_path):
    _seed_alert("X", days_ago=400, db_path=db_path)   # should be deleted
    _seed_alert("X", days_ago=10,  db_path=db_path)   # should survive

    result = audit.prune_old_runs(days_to_keep=30, keep_min_per_symbol=5, db_path=db_path)

    assert _count_alerts(db_path) == 1
    assert result["alerts_deleted"] == 1


def test_prune_no_op_on_empty_db(db_path):
    result = audit.prune_old_runs(days_to_keep=30, keep_min_per_symbol=5, db_path=db_path)
    assert result == {"runs_deleted": 0, "alerts_deleted": 0}


def test_prune_per_symbol_floor_isolated(db_path):
    """The floor is PER SYMBOL — pruning one symbol's history shouldn't affect another."""
    for i in range(20):
        _seed_run("D", days_ago=300 + i, db_path=db_path)   # all old
    for i in range(20):
        _seed_run("E", days_ago=300 + i, db_path=db_path)   # all old

    audit.prune_old_runs(days_to_keep=30, keep_min_per_symbol=5, db_path=db_path)

    with audit._conn(db_path) as c:
        d_count = c.execute("SELECT COUNT(*) FROM analysis_runs WHERE symbol = 'D'").fetchone()[0]
        e_count = c.execute("SELECT COUNT(*) FROM analysis_runs WHERE symbol = 'E'").fetchone()[0]
    assert d_count == 5
    assert e_count == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
