#!/usr/bin/env python3
"""Capital Rotation Monitoring System — long-running entry point.

By default this script runs as a daemon: it reads the `schedule:` block from
`config.yaml`, sleeps until the next scheduled time (default 22:00 in the
configured timezone, weekdays only), fires the daily pipeline, and loops.
SIGINT / SIGTERM trigger a clean shutdown.

Use `--once` to fire a single run and exit (for manual triggers, smoke tests,
or scheduling externally with systemd timers).

Credentials are resolved in strict priority order:

    1. CLI flag         (--telegram-token / --telegram-chat-id / --deepseek-key / --fred-key)
    2. .env file        (KEY=VALUE lines in the script's directory)
    3. fail-fast        (exit 2 with a stderr message naming the missing key)

USAGE:

    # Long-running daemon (default). Fires at the time in config.yaml each weekday.
    python main.py \
        --telegram-token TOKEN --telegram-chat-id ID \
        --deepseek-key KEY --fred-key KEY

    # Same, relying on .env for credentials:
    python main.py

    # One-shot: run pipeline once for today's most recent trading day, then exit.
    python main.py --once

    # One-shot for a specific trading day:
    python main.py --once --date 2026-06-05

The `rotate` CLI still works for ad-hoc operations (rotate fetch / validate /
report). `main.py` is the long-running scheduler.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import date
from pathlib import Path

# Resolve all relative paths against the directory containing this script,
# not the caller's cwd. Cron entries `cd /opt/rotation` first by convention,
# but this also lets `python /opt/rotation/main.py` work from any cwd.
SCRIPT_DIR = Path(__file__).resolve().parent

# Make `import rotation` work without `pip install -e .`. The project uses a
# src/ layout, so we need to put src/ on sys.path. With this, the script-only
# install path (`pip install -r requirements.txt`) is sufficient — no editable
# install needed. `pip install -e .` is still recommended if you want the
# `rotate` CLI entry point as well.
_SRC_DIR = SCRIPT_DIR / "src"
if _SRC_DIR.is_dir() and str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))


# (env var name, --flag, human-readable role) — used for both arg parsing
# and the missing-credential error message.
REQUIRED_KEYS: list[tuple[str, str, str]] = [
    ("TELEGRAM_BOT_TOKEN",  "--telegram-token",    "Telegram bot token"),
    ("TELEGRAM_CHAT_ID",    "--telegram-chat-id",  "Telegram chat id"),
    ("DEEPSEEK_API_KEY",    "--deepseek-key",      "DeepSeek API key (LLM Section 12)"),
    ("FRED_API_KEY",        "--fred-key",          "FRED API key (RRP / yield curve)"),
]


def load_dotenv(path: str | Path = ".env") -> dict[str, str]:
    """Read `KEY=VALUE` lines from a .env file. Strips surrounding quotes.
    Relative paths resolve against the script's directory (not cwd).
    Returns an empty dict if no file."""
    p = Path(path)
    if not p.is_absolute():
        p = SCRIPT_DIR / p
    if not p.exists():
        return {}
    out: dict[str, str] = {}
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key:
            out[key] = val
    return out


def resolve_credentials(
    cli_args: dict[str, str | None],
    dotenv: dict[str, str],
) -> tuple[dict[str, str], list[tuple[str, str, str]]]:
    """Apply the strict priority order. Returns (resolved, missing).

    `resolved` contains the keys that were found, mapped to their values.
    `missing` is a list of (env_name, flag_name, role) tuples for keys that
    weren't found anywhere. The caller decides what to do with missing keys
    (main.py exits; tests inspect the list).
    """
    resolved: dict[str, str] = {}
    missing: list[tuple[str, str, str]] = []
    for env_name, flag_name, role in REQUIRED_KEYS:
        value = cli_args.get(env_name)            # 1. CLI flag wins
        if not value:
            value = dotenv.get(env_name)          # 2. .env fallback
        if not value:
            missing.append((env_name, flag_name, role))
        else:
            resolved[env_name] = value
    return resolved, missing


def _format_missing_error(missing: list[tuple[str, str, str]]) -> str:
    lines = [
        "ERROR: Missing required credentials. Provide each via a CLI flag",
        "       or by setting it in `.env` (one KEY=VALUE per line).",
        "",
    ]
    for env_name, flag_name, role in missing:
        lines.append(f"  - {env_name:22s} ({flag_name})  — {role}")
    lines.append("")
    lines.append("Run aborted (exit code 2). No part of the pipeline executed.")
    lines.append("")
    return "\n".join(lines)


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="main.py",
        description="Long-running scheduler for the Capital Rotation daily pipeline.",
    )
    p.add_argument(
        "--once", action="store_true",
        help="Run the pipeline once and exit (no scheduler loop).",
    )
    p.add_argument(
        "--date", default=None,
        help="YYYY-MM-DD. Implies --once. Defaults to today (snapped to last NYSE trading day).",
    )
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml.")
    p.add_argument("--telegram-token",    dest="TELEGRAM_BOT_TOKEN", default=None)
    p.add_argument("--telegram-chat-id",  dest="TELEGRAM_CHAT_ID",   default=None)
    p.add_argument("--deepseek-key",      dest="DEEPSEEK_API_KEY",   default=None)
    p.add_argument("--fred-key",          dest="FRED_API_KEY",       default=None)
    return p


def main(argv: list[str] | None = None) -> int:
    # Windows consoles often default to cp1252, which can't encode the ✓/✗
    # marks in the summary output. Reconfigure rather than crash.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    args = _build_arg_parser().parse_args(argv)

    cli_args = {
        env_name: getattr(args, env_name, None)
        for env_name, _, _ in REQUIRED_KEYS
    }
    dotenv = load_dotenv()
    resolved, missing = resolve_credentials(cli_args, dotenv)

    if missing:
        sys.stderr.write(_format_missing_error(missing))
        return 2

    # Promote to env so the rotation package's existing readers (alerts.py,
    # llm_interpret.py, ingest/fred_adapter.py) pick them up transparently.
    for k, v in resolved.items():
        os.environ[k] = v

    # Lazy imports — must come AFTER env is populated, because some modules
    # capture state at import time.
    import threading
    from rotation.config import load_config
    from rotation.pipeline import run_daily
    from rotation.scheduler import (
        SchedulerState, install_signal_handlers, run_forever,
    )

    # Make config path script-relative if the caller didn't provide an absolute one.
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = SCRIPT_DIR / config_path

    # cd into the script's directory so the rotation package's relative writes
    # (data/, reports/, alerts/) land in the project tree regardless of cwd.
    os.chdir(SCRIPT_DIR)
    cfg = load_config(config_path)

    # A row stuck at status='running' means a prior run crashed mid-pipeline
    # (log_run_finish never executed). Data integrity is unaffected — upserts
    # are transactional — but the operator should know the day may be partial.
    from rotation.store import connect
    with connect(cfg.storage.duckdb_path) as con:
        orphans = con.execute(
            "SELECT run_id, asof_date FROM run_log "
            "WHERE status = 'running' ORDER BY started_at DESC"
        ).fetchall()
    if orphans:
        sys.stderr.write(
            f"WARNING: {len(orphans)} prior run(s) never finished (crashed mid-run?):\n"
        )
        for run_id, asof_date in orphans[:5]:
            sys.stderr.write(f"  - {run_id} (asof {asof_date})\n")
        if len(orphans) > 5:
            sys.stderr.write(f"  … and {len(orphans) - 5} more\n")

    # --date implies --once
    one_shot = args.once or bool(args.date)

    if one_shot:
        target = date.fromisoformat(args.date) if args.date else date.today()
        summary = run_daily(cfg, target)
        _print_summary(summary)
        return 0 if summary.get("ok") else 1

    # Long-running scheduler mode.
    if cfg.schedule is None:
        sys.stderr.write(
            "ERROR: schedule mode requires a `schedule:` block in config.yaml.\n"
            "       Either add one (see config.yaml's example) or pass --once.\n"
        )
        return 2

    sched_cfg = cfg.schedule
    print(
        f"Scheduler started. Firing daily at {sched_cfg.daily_at} ({sched_cfg.timezone})"
        f"{' weekdays only' if sched_cfg.weekdays_only else ''}"
        f"{'; run-on-startup' if sched_cfg.run_on_startup else ''}"
        f"{'; catch-up missed' if sched_cfg.catch_up_missed else ''}.",
        flush=True,
    )
    print("SIGINT or SIGTERM exits cleanly.\n", flush=True)

    def _job():
        summary = run_daily(cfg, date.today())
        _print_summary(summary)

    state = SchedulerState(stop_event=threading.Event())
    install_signal_handlers(state)
    run_forever(sched_cfg, _job, state=state)
    return 0


def _print_summary(summary: dict) -> None:
    asof = summary.get("asof", "?")
    status = "OK" if summary.get("ok") else "FAILED"
    print(f"\nPipeline {status}  asof={asof}")
    for step in summary.get("steps", []):
        mark = "✓" if step.get("ok") else "✗"
        dur = step.get("duration_s", 0.0)
        suffix = "" if step.get("ok") else f"  {step.get('error', '')}"
        print(f"  {mark} {step['step']:9s} [{dur:5.1f}s]{suffix}")
    if summary.get("heartbeat"):
        print(f"  heartbeat:      {summary['heartbeat']}")
    for key in ("pdf_attachment_us", "pdf_attachment_hk"):
        if summary.get(key):
            print(f"  {key}: {summary[key]}")


if __name__ == "__main__":
    sys.exit(main())
