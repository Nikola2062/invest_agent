"""Manual one-shot trigger — useful for testing a push without waiting for cron.

Usage:
    .venv/bin/python -m src.notifier.run_once             # runs us_pre_open by default
    .venv/bin/python -m src.notifier.run_once hk_pre_close
"""
from __future__ import annotations

import argparse
import logging
import sys

from src.config.loader import load_schedule
from src.notifier.scheduler import _job


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
    parser = argparse.ArgumentParser(description="Manually trigger a Telegram push.")
    parser.add_argument(
        "push_name",
        nargs="?",
        default="us_pre_open",
        help="Push name from config/schedule.yaml (e.g. hk_morning_recap, hk_pre_close, us_pre_open, us_pre_close)",
    )
    args = parser.parse_args()

    sched = load_schedule()
    target = next((p for p in sched.pushes if p.name == args.push_name), None)
    if target is None:
        print(f"Push {args.push_name!r} not found. Available: {[p.name for p in sched.pushes]}", file=sys.stderr)
        return 2

    _job(target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
