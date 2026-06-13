"""Tiny in-process scheduler for the daily pipeline.

No external dependency — just stdlib `zoneinfo` (Python ≥ 3.9) + signals +
threading.Event. The intended deployment is a long-lived `python main.py`
process; this module computes the next fire time, sleeps in interruptible
chunks (so SIGTERM unblocks immediately), and yields control back to the
caller when the time arrives.

The caller is responsible for actually running the pipeline. This module
just answers "when next" and provides the sleep loop.
"""
from __future__ import annotations

import logging
import signal
import threading
from dataclasses import dataclass
from datetime import datetime, time, timedelta, tzinfo
from typing import Callable
from zoneinfo import ZoneInfo

from .config import ScheduleConfig

log = logging.getLogger(__name__)


WEEKDAYS = {0, 1, 2, 3, 4}     # Mon=0, Sun=6 in datetime.weekday()


def _parse_hhmm(s: str) -> time:
    """Accepts '22:00' or '22:00:00'."""
    parts = s.split(":")
    if len(parts) == 2:
        h, m = int(parts[0]), int(parts[1])
        return time(hour=h, minute=m)
    if len(parts) == 3:
        h, m, sec = int(parts[0]), int(parts[1]), int(parts[2])
        return time(hour=h, minute=m, second=sec)
    raise ValueError(f"daily_at must be HH:MM or HH:MM:SS, got: {s!r}")


def next_run_at(
    sched: ScheduleConfig,
    now: datetime,
    catch_up_missed: bool | None = None,
) -> datetime:
    """Return the next datetime (TZ-aware) at which the pipeline should fire.

    Logic:
      - Compute today's scheduled fire-time in the configured TZ.
      - If `now` is already past that and catch_up_missed is True, return now
        (fire immediately). Otherwise, move forward.
      - If weekdays_only and the target lands on Sat/Sun, advance to Monday.

    `now` should be timezone-aware. Pass `now=datetime.now(ZoneInfo("UTC"))`
    or any other tz; we convert internally.
    """
    tz = ZoneInfo(sched.timezone)
    t = _parse_hhmm(sched.daily_at)
    catch_up = sched.catch_up_missed if catch_up_missed is None else catch_up_missed

    now_local = now.astimezone(tz)
    target_today = datetime.combine(now_local.date(), t, tzinfo=tz)

    if now_local <= target_today:
        candidate = target_today
    else:
        if catch_up and (not sched.weekdays_only or now_local.weekday() in WEEKDAYS):
            return now_local
        candidate = target_today + timedelta(days=1)

    while sched.weekdays_only and candidate.weekday() not in WEEKDAYS:
        candidate += timedelta(days=1)

    return candidate


@dataclass
class SchedulerState:
    """Lightweight handle so callers (and tests) can stop the loop cleanly."""
    stop_event: threading.Event

    def stop(self) -> None:
        self.stop_event.set()


def install_signal_handlers(state: SchedulerState) -> None:
    """SIGINT and SIGTERM both set the stop event so the loop exits at the
    next sleep wake-up (within `tick_seconds`). Cron orchestrators (systemd,
    Docker) send SIGTERM on shutdown; we want a clean exit, not a SIGKILL."""
    def _handler(signum, _frame):
        log.info("scheduler: received signal %d, requesting stop", signum)
        state.stop()
    signal.signal(signal.SIGTERM, _handler)
    signal.signal(signal.SIGINT, _handler)


def run_forever(
    sched: ScheduleConfig,
    job: Callable[[], None],
    tick_seconds: float = 30.0,
    now_fn: Callable[[], datetime] | None = None,
    state: SchedulerState | None = None,
) -> None:
    """Block the calling thread, firing `job()` at each scheduled time.

    `tick_seconds` is the sleep granularity — shorter = faster shutdown
    response to signals, longer = lower idle CPU. 30s is fine for daily.

    `now_fn` is injectable for testing (lets tests advance virtual time).
    """
    if now_fn is None:
        def now_fn():
            return datetime.now(ZoneInfo(sched.timezone))
    state = state or SchedulerState(stop_event=threading.Event())

    if sched.run_on_startup:
        log.info("scheduler: run_on_startup=true; firing job immediately")
        _safe_run(job)
    elif not sched.catch_up_missed:
        # Surface silently-skipped runs: if the process starts after today's
        # fire time, catch_up_missed=false means today's run will NOT happen.
        # Without this line the operator only notices via a missing heartbeat.
        now = now_fn()
        if next_run_at(sched, now, catch_up_missed=True) <= now:
            log.warning(
                "scheduler: started after today's %s (%s) fire time and "
                "catch_up_missed=false — today's run is SKIPPED; run "
                "`python main.py --once` manually if the day must not be missed",
                sched.daily_at, sched.timezone,
            )

    while not state.stop_event.is_set():
        target = next_run_at(sched, now_fn())
        log.info("scheduler: next fire at %s (%s)",
                 target.isoformat(), sched.timezone)
        # Sleep in short ticks so SIGTERM unblocks fast.
        while not state.stop_event.is_set():
            remaining = (target - now_fn()).total_seconds()
            if remaining <= 0:
                break
            state.stop_event.wait(timeout=min(tick_seconds, remaining))
        if state.stop_event.is_set():
            break
        _safe_run(job)
        # Loop continues; next_run_at() will return tomorrow's time.

    log.info("scheduler: loop exited cleanly")


def _safe_run(job: Callable[[], None]) -> None:
    """Run the job, log + swallow exceptions so one bad day doesn't kill the
    long-running process. The job itself (run_daily) emits its own Telegram
    heartbeat with failure detail, so callers see what broke without needing
    to scrape stderr."""
    try:
        job()
    except Exception:
        log.exception("scheduler: job raised; loop continues (next fire tomorrow)")
