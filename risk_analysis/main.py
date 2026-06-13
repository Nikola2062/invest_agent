"""Entry point.

Telegram delivery is controlled by `telegram.enabled` in config.json.
When enabled, --bot-token and --chat-id MUST be passed as CLI args
(credentials are never read from disk).

Examples
--------
config.json -> telegram.enabled = false:
    python main.py                # run once, save JSON only
    python main.py --schedule     # daemonize

config.json -> telegram.enabled = true:
    python main.py --bot-token <T> --chat-id <C>
    python main.py --schedule --bot-token <T> --chat-id <C>
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import schedule

from analyzer import build_report, format_telegram
from notifier import send_telegram

ROOT = Path(__file__).parent
CONFIG_PATH = ROOT / "config.json"
ENV_PATH = ROOT / ".env"


def _load_dotenv(path):
    """Minimal .env loader (no external dependency).

    Lines look like KEY=value; blank lines and # comments are ignored.
    Real environment variables win (setdefault), so the shell can override .env.
    """
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def load_config():
    _load_dotenv(ENV_PATH)
    with open(CONFIG_PATH) as f:
        config = json.load(f)
    # Secret never lives in config.json: pull it from the environment (.env).
    # An explicit non-empty value in config.json still wins, for back-compat.
    if not config.get("fred_api_key"):
        config["fred_api_key"] = os.environ.get("FRED_API_KEY", "")
    return config


def save_report(report, reports_dir):
    reports_dir = Path(reports_dir)
    if not reports_dir.is_absolute():
        reports_dir = ROOT / reports_dir
    reports_dir.mkdir(parents=True, exist_ok=True)
    # Filename includes HHMM so multiple daily runs don't overwrite each other.
    hhmm = datetime.now().strftime("%H%M")
    out = reports_dir / f"{report['date']}_{hhmm}.json"
    with open(out, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return out


def run_once(telegram=None):
    """Build a report, save it, optionally push to Telegram.

    telegram: None to skip sending, or a (bot_token, chat_id) tuple to send.
    """
    config = load_config()
    print(f"[{datetime.now().isoformat(timespec='seconds')}] Building report…")
    report = build_report(config)
    saved = save_report(report, config.get("reports_dir", "reports"))
    print(f"  saved → {saved}")

    if telegram is None:
        print("  ⏭  Telegram disabled (telegram.enabled = false).")
        return

    msg = format_telegram(report)
    bot_token, chat_id = telegram
    try:
        send_telegram(bot_token, chat_id, msg)
        print("  ✓ sent to Telegram")
    except Exception as e:
        print(f"  ✗ Telegram send failed: {e}")


def run_scheduled(telegram=None):
    config = load_config()
    sched = config.get("schedule", {})
    if not sched.get("enabled", True):
        print("Scheduler disabled in config. Exiting.")
        return
    # Accept either `times: ["09:25", "12:00"]` (preferred) or legacy `time: "09:25"`.
    times = sched.get("times") or ([sched["time"]] if sched.get("time") else [])
    if not times:
        print("No schedule times configured. Set schedule.times in config.json. Exiting.")
        return
    # Times are interpreted in the configured timezone; defaults to US Eastern
    # so HH:MM entries map to the US trading session and DST is auto-handled.
    tz = sched.get("timezone", "America/New_York")
    for t in times:
        schedule.every().day.at(t, tz).do(run_once, telegram=telegram)
    tg_state = "ON" if telegram else "OFF"
    print(f"Scheduler armed. Daily runs at {', '.join(times)} ({tz}). Telegram: {tg_state}.")
    while True:
        schedule.run_pending()
        time.sleep(30)


def resolve_telegram(config, args, parser):
    """Return (bot_token, chat_id) when enabled, else None. Exit on misuse."""
    enabled = bool(config.get("telegram", {}).get("enabled", False))
    if enabled:
        if not (args.bot_token and args.chat_id):
            parser.error(
                "config.json -> telegram.enabled is true, so --bot-token "
                "and --chat-id are required to start the program."
            )
        return (args.bot_token, args.chat_id)
    if args.bot_token or args.chat_id:
        parser.error(
            "config.json -> telegram.enabled is false; remove --bot-token / "
            "--chat-id or flip telegram.enabled to true."
        )
    return None


def parse_args(argv):
    p = argparse.ArgumentParser(description="US equity crash-risk notifier.")
    p.add_argument("--schedule", action="store_true",
                   help="Run as a daemon firing at config.json schedule.times.")
    p.add_argument("--bot-token",
                   help="Telegram bot token (required when telegram.enabled is true).")
    p.add_argument("--chat-id",
                   help="Telegram chat id (required when telegram.enabled is true).")
    return p, p.parse_args(argv)


if __name__ == "__main__":
    parser, args = parse_args(sys.argv[1:])
    telegram = resolve_telegram(load_config(), args, parser)
    if args.schedule:
        run_scheduled(telegram=telegram)
    else:
        run_once(telegram=telegram)
