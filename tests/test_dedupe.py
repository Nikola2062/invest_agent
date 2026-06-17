"""Unit tests for the per-name deep-dive dedupe (``positions.select_for_deepdive``).

Pure function — state and `now` are passed in as values, so no clock or disk.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from tradingagents.portfolio.positions import DedupeConfig, select_for_deepdive

pytestmark = pytest.mark.unit

CFG = DedupeConfig()  # rerun_worsen_pct=5, rerun_after_days=5
NOW = datetime(2026, 6, 12, tzinfo=timezone.utc)


def _alarm(sym="FIG", dd=-20.0, cats=("drawdown",)):
    return {"symbol": sym, "reasons": [f"{dd:+.0f}% from high"],
            "categories": list(cats), "drawdown": dd}


def _state(sym="FIG", dd=-20.0, cats=("drawdown",), ts="2026-06-10T00:00:00+00:00"):
    return {sym: {"ts": ts, "drawdown": dd, "categories": list(cats)}}


def _syms(alarms):
    return {a["symbol"] for a in alarms}


# --------------------------------- new name ----------------------------------

def test_unseen_name_runs():
    to_run, new_state = select_for_deepdive([_alarm()], {}, NOW, CFG)
    assert _syms(to_run) == {"FIG"}
    assert new_state["FIG"]["ts"] == NOW.isoformat()


# ------------------------------ steady state ---------------------------------

def test_unchanged_within_cooldown_is_deduped():
    to_run, new_state = select_for_deepdive([_alarm(dd=-20.0)], _state(dd=-20.0), NOW, CFG)
    assert to_run == []
    # prior record carried forward unchanged (cooldown anchored to last RUN)
    assert new_state["FIG"]["ts"] == "2026-06-10T00:00:00+00:00"


# ---------------------------- material change --------------------------------

def test_new_category_reruns():
    alarm = _alarm(cats=("drawdown", "risk_off"))
    to_run, _ = select_for_deepdive([alarm], _state(cats=("drawdown",)), NOW, CFG)
    assert _syms(to_run) == {"FIG"}


def test_drawdown_worsening_past_threshold_reruns():
    # -20 -> -26 is 6 pts worse (>= 5)
    to_run, _ = select_for_deepdive([_alarm(dd=-26.0)], _state(dd=-20.0), NOW, CFG)
    assert _syms(to_run) == {"FIG"}


def test_drawdown_worsening_below_threshold_is_deduped():
    # -20 -> -23 is only 3 pts worse (< 5)
    to_run, _ = select_for_deepdive([_alarm(dd=-23.0)], _state(dd=-20.0), NOW, CFG)
    assert to_run == []


# -------------------------------- staleness ----------------------------------

def test_stale_reruns_after_interval():
    later = datetime(2026, 6, 20, tzinfo=timezone.utc)   # 10 days > 5
    to_run, new_state = select_for_deepdive([_alarm()], _state(), later, CFG)
    assert _syms(to_run) == {"FIG"}
    assert new_state["FIG"]["ts"] == later.isoformat()


def test_tz_naive_prev_treated_as_stale():
    naive = _state(ts="2026-06-10T00:00:00")             # no tz -> can't subtract
    to_run, _ = select_for_deepdive([_alarm()], naive, NOW, CFG)
    assert _syms(to_run) == {"FIG"}


# ------------------------------- recovery ------------------------------------

def test_recovered_name_dropped_from_state():
    # Prior state had FIG + AMD; only FIG still alarms -> AMD pruned.
    prev = {**_state("FIG"), **_state("AMD")}
    _, new_state = select_for_deepdive([_alarm("FIG")], prev, NOW, CFG)
    assert set(new_state.keys()) == {"FIG"}


# --------------------------------- config ------------------------------------

def test_custom_worsen_threshold():
    loose = DedupeConfig(rerun_worsen_pct=2.0)
    to_run, _ = select_for_deepdive([_alarm(dd=-23.0)], _state(dd=-20.0), NOW, loose)
    assert _syms(to_run) == {"FIG"}
