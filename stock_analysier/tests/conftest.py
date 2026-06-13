"""Shared pytest fixtures. Isolates tests from production audit DB.

`tactical_exit.decide_tactical` reads `src.storage.audit.get_recent_runs` to
enforce persistence-days. Without isolation, real prior runs against the same
symbol contaminate test outcomes. We redirect every audit accessor to a
per-test temp DB by overriding the `db_path=DEFAULT_DB_PATH` default that gets
baked into each function at definition time.
"""
from __future__ import annotations

import functools

import pytest

from src.storage import audit


_FUNCS_WITH_DB_PATH = (
    "record_analysis",
    "get_recent_runs",
    "get_latest_run",
    "get_previous_run_before",
    "record_alert",
    "mark_alert_pushed",
    "get_unpushed_alerts",
    "get_recent_alerts",
    "count_recent_runs_with_signal",
)


@pytest.fixture(autouse=True)
def _isolated_audit_db(tmp_path_factory, monkeypatch):
    test_db = tmp_path_factory.mktemp("audit") / "audit.sqlite"
    monkeypatch.setattr(audit, "DEFAULT_DB_PATH", test_db)

    for name in _FUNCS_WITH_DB_PATH:
        original = getattr(audit, name)

        def _make_wrapper(fn):
            @functools.wraps(fn)
            def wrapper(*args, **kwargs):
                kwargs.setdefault("db_path", test_db)
                return fn(*args, **kwargs)
            return wrapper

        monkeypatch.setattr(audit, name, _make_wrapper(original))

    yield test_db
