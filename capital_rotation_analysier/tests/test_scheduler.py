"""Tests for the scheduler module — next_run_at semantics, weekday skip,
DST handling, and the run_forever loop under virtual time."""
from __future__ import annotations

import threading
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pytest

from rotation.config import ScheduleConfig
from rotation.scheduler import (
    SchedulerState, _parse_hhmm, next_run_at, run_forever,
)


def _sched(daily_at="22:00", tz="UTC", weekdays_only=True,
           run_on_startup=False, catch_up_missed=False) -> ScheduleConfig:
    return ScheduleConfig(
        daily_at=daily_at, timezone=tz, weekdays_only=weekdays_only,
        run_on_startup=run_on_startup, catch_up_missed=catch_up_missed,
    )


# ---------- _parse_hhmm ----------

def test_parse_hhmm_basic():
    t = _parse_hhmm("22:30")
    assert (t.hour, t.minute, t.second) == (22, 30, 0)


def test_parse_hhmm_with_seconds():
    t = _parse_hhmm("09:00:45")
    assert (t.hour, t.minute, t.second) == (9, 0, 45)


def test_parse_hhmm_rejects_garbage():
    with pytest.raises(ValueError):
        _parse_hhmm("not a time")


# ---------- next_run_at ----------

UTC = ZoneInfo("UTC")
NY  = ZoneInfo("America/New_York")


def test_next_run_at_today_if_target_is_in_the_future():
    # Wednesday 2026-06-03 at 10:00 UTC. Target 22:00 today → same day.
    now = datetime(2026, 6, 3, 10, 0, tzinfo=UTC)
    nxt = next_run_at(_sched(daily_at="22:00", tz="UTC"), now)
    assert nxt == datetime(2026, 6, 3, 22, 0, tzinfo=UTC)


def test_next_run_at_tomorrow_if_target_already_passed():
    # Wed 23:00 UTC, target 22:00 → tomorrow (Thursday) 22:00.
    now = datetime(2026, 6, 3, 23, 0, tzinfo=UTC)
    nxt = next_run_at(_sched(), now)
    assert nxt == datetime(2026, 6, 4, 22, 0, tzinfo=UTC)


def test_next_run_at_weekdays_only_skips_saturday_and_sunday():
    # Friday 23:00 UTC → next weekday is Monday 22:00.
    now = datetime(2026, 6, 5, 23, 0, tzinfo=UTC)  # Friday
    nxt = next_run_at(_sched(weekdays_only=True), now)
    # Monday 2026-06-08 at 22:00 UTC
    assert nxt == datetime(2026, 6, 8, 22, 0, tzinfo=UTC)
    assert nxt.weekday() == 0  # Mon


def test_next_run_at_allows_weekends_when_disabled():
    now = datetime(2026, 6, 5, 23, 0, tzinfo=UTC)  # Friday
    nxt = next_run_at(_sched(weekdays_only=False), now)
    # Should be Saturday at 22:00
    assert nxt == datetime(2026, 6, 6, 22, 0, tzinfo=UTC)


def test_next_run_at_catch_up_returns_now_when_past_target():
    # Past target same day, catch_up enabled, weekday → return now.
    now = datetime(2026, 6, 3, 23, 0, tzinfo=UTC)  # Wed
    nxt = next_run_at(_sched(catch_up_missed=True), now)
    assert nxt == now


def test_next_run_at_catch_up_still_skips_weekend():
    # Sat 23:00 with catch_up enabled but weekdays_only on → Monday.
    now = datetime(2026, 6, 6, 23, 0, tzinfo=UTC)  # Saturday
    nxt = next_run_at(_sched(weekdays_only=True, catch_up_missed=True), now)
    assert nxt.weekday() == 0
    assert nxt == datetime(2026, 6, 8, 22, 0, tzinfo=UTC)


def test_next_run_at_timezone_independent_of_caller_tz():
    # Provide `now` in UTC, schedule in NY tz, expect target in NY tz.
    now_utc = datetime(2026, 6, 3, 14, 0, tzinfo=UTC)  # Wed 10am ET
    sched = _sched(daily_at="16:30", tz="America/New_York")
    nxt = next_run_at(sched, now_utc)
    # 16:30 ET on the same day
    expected_ny = datetime(2026, 6, 3, 16, 30, tzinfo=NY)
    assert nxt == expected_ny


def test_next_run_at_dst_spring_forward():
    """DST transitions: scheduled in NY tz, fire time stays at the same wall-clock
    hour across the spring-forward boundary."""
    sched = _sched(daily_at="22:00", tz="America/New_York")
    # Day before DST starts (2026-03-08 in the US). Fire at 22:00 EST.
    before = datetime(2026, 3, 7, 12, 0, tzinfo=NY)  # Saturday before DST
    nxt_before = next_run_at(_sched(daily_at="22:00", tz="America/New_York",
                                    weekdays_only=False), before)
    # Day during DST. 22:00 EDT.
    during = datetime(2026, 3, 9, 12, 0, tzinfo=NY)  # Monday after DST
    nxt_during = next_run_at(sched, during)
    # Wall-clock hour preserved in both
    assert nxt_before.hour == 22
    assert nxt_during.hour == 22


# ---------- run_forever ----------
#
# These use a clock that advances on every now_fn() call. The scheduler
# checks now_fn() in the inner wait loop; after enough virtual ticks the
# target time passes and the job fires. Tiny real-time, deterministic.

def _advancing_clock(start: datetime, step_seconds: int = 5):
    clock = [start]
    def now_fn():
        v = clock[0]
        clock[0] = v + timedelta(seconds=step_seconds)
        return v
    return now_fn


def test_run_forever_fires_at_scheduled_time_with_virtual_clock():
    sched = _sched(daily_at="22:00", tz="UTC", weekdays_only=False)
    state = SchedulerState(stop_event=threading.Event())
    fire_count = {"n": 0}

    # Clock starts 30 seconds before fire; each call advances 5 seconds.
    # Sequence: 21:59:30 → 21:59:35 → ... → 22:00:00 → fire.
    now_fn = _advancing_clock(
        start=datetime(2026, 6, 3, 21, 59, 30, tzinfo=UTC),
        step_seconds=5,
    )

    def job():
        fire_count["n"] += 1
        state.stop()  # one-shot for this test

    run_forever(sched, job, tick_seconds=0.01, now_fn=now_fn, state=state)
    assert fire_count["n"] == 1


def test_run_forever_run_on_startup_fires_before_first_sleep():
    sched = _sched(run_on_startup=True, weekdays_only=False)
    state = SchedulerState(stop_event=threading.Event())
    fire_count = {"n": 0}

    def now_fn():
        return datetime(2026, 6, 3, 10, 0, tzinfo=UTC)

    def job():
        fire_count["n"] += 1
        state.stop()

    run_forever(sched, job, tick_seconds=0.01, now_fn=now_fn, state=state)
    assert fire_count["n"] == 1


def test_run_forever_swallows_job_exception_and_continues():
    """One bad day shouldn't kill the long-running process."""
    sched = _sched(run_on_startup=True, weekdays_only=False)
    state = SchedulerState(stop_event=threading.Event())
    calls = {"n": 0}

    def now_fn():
        return datetime(2026, 6, 3, 10, 0, tzinfo=UTC)

    def job():
        calls["n"] += 1
        state.stop()
        raise RuntimeError("simulated failure")

    run_forever(sched, job, tick_seconds=0.01, now_fn=now_fn, state=state)
    assert calls["n"] == 1


def test_scheduler_state_stop_unblocks_wait():
    """SIGTERM-style stop must wake the wait() and exit cleanly within
    tick_seconds. If stop is broken, this test would hang on the 12-hour gap."""
    sched = _sched(daily_at="22:00", tz="UTC", weekdays_only=False)
    state = SchedulerState(stop_event=threading.Event())
    started = threading.Event()

    def now_fn():
        started.set()
        return datetime(2026, 6, 3, 10, 0, tzinfo=UTC)  # 12h until fire

    def job():
        pytest.fail("job should not fire; stop should exit first")

    def _stop_after_short_delay():
        started.wait(timeout=2)
        state.stop()

    t = threading.Thread(target=_stop_after_short_delay, daemon=True)
    t.start()
    run_forever(sched, job, tick_seconds=0.05, now_fn=now_fn, state=state)
    t.join(timeout=1)
