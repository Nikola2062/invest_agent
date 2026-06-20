"""Holiday-aware, timezone-aware pre-market scheduler.

Fires a callback at each configured (time, timezone) — 1h before each market open.
Each run is evaluated in its OWN timezone (DST handled via zoneinfo), fires at most
once per local calendar day, and is HOLIDAY-aware when pandas-market-calendars is
installed (optional); otherwise it falls back to a weekday filter.

Ported from investor_agent/orchestrator (was _run_scheduler in its main.py), made
dependency-light: no apscheduler, no hard pandas-market-calendars requirement.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo


def _trading_day(cal, d) -> bool:
    return not cal.schedule(start_date=str(d), end_date=str(d)).empty


def run_scheduler(runs: list[dict], weekdays_only: bool, on_fire: Callable[[], None],
                  run_now: bool = False, poll_seconds: int = 20) -> None:
    """Block forever, firing ``on_fire`` at each run's local time.

    ``runs`` items: {label, time "HH:MM", timezone, calendar?}. ``on_fire`` takes no
    args and should build+send the digest; exceptions are logged, never fatal.
    """
    try:
        import pandas_market_calendars as mcal
    except Exception:
        mcal = None

    zones = []  # (run, tzinfo, calendar_or_None)
    for r in runs:
        try:
            tz = ZoneInfo(r["timezone"])
        except Exception as e:
            print(f"[scheduler] bad timezone {r.get('timezone')!r}: {e}", file=sys.stderr)
            continue
        cal = None
        if r.get("calendar") and mcal is not None:
            try:
                cal = mcal.get_calendar(r["calendar"])
            except Exception as e:
                print(f"[scheduler] bad calendar {r['calendar']!r} for '{r.get('label')}': {e} "
                      f"(falling back to weekdays_only)", file=sys.stderr)
        zones.append((r, tz, cal))

    desc = ", ".join(
        f"{r['label']} {r['time']} {r['timezone']}"
        f"{' [' + r['calendar'] + ']' if cal is not None else ' [weekdays]'}"
        for r, _, cal in zones
    )
    print(f"[scheduler] runs: {desc}", file=sys.stderr)

    if run_now:
        try:
            on_fire()
            print("[scheduler] fired at startup (--run-now)", file=sys.stderr)
        except Exception as e:
            print(f"[scheduler] startup fire failed: {e}", file=sys.stderr)

    last_fired: dict = {}  # label -> local date already handled
    while True:
        for r, tz, cal in zones:
            now = datetime.now(tz)
            label = r["label"]
            if now.strftime("%H:%M") != r["time"] or last_fired.get(label) == now.date():
                continue
            last_fired[label] = now.date()
            if cal is not None:
                if not _trading_day(cal, now.date()):
                    print(f"[scheduler] '{label}' {now.date()} — {r['calendar']} closed, skipped",
                          file=sys.stderr)
                    continue
            elif weekdays_only and now.weekday() >= 5:
                print(f"[scheduler] '{label}' {now.date()} — weekend, skipped", file=sys.stderr)
                continue
            try:
                on_fire()
                print(f"[scheduler] fired '{label}' {now.isoformat(timespec='seconds')}", file=sys.stderr)
            except Exception as e:
                print(f"[scheduler] '{label}' fire failed: {e}", file=sys.stderr)
        time.sleep(poll_seconds)
