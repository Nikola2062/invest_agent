"""APScheduler-driven Telegram pusher.

Loads config/schedule.yaml, registers a cron trigger per push, each job:
  1. Runs the full pipeline on portfolio + watchlist
  2. Formats a brief digest filtered to the push's market
  3. Sends via Telegram

Run with:
    .venv/bin/python -m src.notifier.scheduler
or
    ./run_scheduler.sh
"""
from __future__ import annotations

import logging
import signal
import sys
import time

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.agents.monitor import check_holdings
from src.config.loader import load_portfolio, load_schedule
from src.models.schemas import AnalysisResult, PushConfig
from src.notifier.formatter import format_brief_digest, format_thesis_break_alerts
from src.notifier.telegram_client import TelegramError, send_message
from src.pipeline.orchestrator import analyze_all, reset_macro_cache
from src.storage.audit import get_unpushed_alerts, mark_alert_pushed

log = logging.getLogger(__name__)


def _has_actionable_signal(
    results: list[AnalysisResult],
    market_filter: str | None,
    alerts_pending: int,
) -> tuple[bool, str]:
    """Returns (actionable, reason). market_filter=None or 'ALL' = no filter."""
    if alerts_pending > 0:
        return True, f"{alerts_pending} pending thesis-break alert(s)"
    portfolio = load_portfolio()
    held_symbols = {h.symbol for h in portfolio.holdings}
    filt = (market_filter not in (None, "", "ALL"))
    for r in results:
        if filt and r.market != market_filter:
            continue
        if r.symbol in held_symbols and r.if_held.tactical.label:
            return True, f"{r.symbol}: tactical {r.if_held.tactical.label}"
        if r.symbol not in held_symbols and r.if_not_held.recommendation in ("BUY_NOW", "WAIT_FOR_PRICE"):
            return True, f"{r.symbol}: {r.if_not_held.recommendation}"
        da = getattr(r, "devil_advocate", None)
        if da and da.overall_verdict == "veto" and r.symbol in held_symbols:
            return True, f"{r.symbol}: DA veto on held position"
    return False, "no held tactical action, no watchlist BUY_NOW/WAIT, no pending alerts"


def _job(push: PushConfig) -> None:
    log.info("Push job firing: %s (%s)", push.name, push.purpose)
    try:
        # 1. Refresh macro + run analysis (this persists to audit trail)
        reset_macro_cache()
        results = analyze_all()

        # 2. Monitor: compare latest 2 audit rows per holding, generate alerts
        try:
            new_breaks = check_holdings()
            if new_breaks:
                log.info("Monitor: %d new thesis-break alerts.", len(new_breaks))
        except Exception as e:
            log.warning("Monitor check failed: %s", e)

        # 3. Send out-of-band alert digest if any unpushed alerts exist
        # (always sent — these are by definition things the user must know about)
        unpushed = get_unpushed_alerts()
        if unpushed:
            alert_text = format_thesis_break_alerts(unpushed)
            try:
                send_message(alert_text)
                for row in unpushed:
                    mark_alert_pushed(row["id"])
                log.info("Pushed %d thesis-break alerts.", len(unpushed))
            except TelegramError as e:
                log.error("Alert push failed: %s", e)
            alerts_pushed_now = len(unpushed)
        else:
            alerts_pushed_now = 0

        # 4. Send the scheduled brief digest — optionally suppressed on quiet days
        market_filter = None if push.market in ("", "ALL") else push.market
        if push.send_only_if_action:
            actionable, reason = _has_actionable_signal(results, market_filter, alerts_pushed_now)
            if not actionable:
                log.info("Push %s: SUPPRESSED — %s (send_only_if_action=true).", push.name, reason)
                return
            log.info("Push %s: actionable trigger — %s", push.name, reason)

        digest = format_brief_digest(
            results,
            push_name=push.name,
            market_filter=market_filter,
            purpose=push.purpose,
        )
        send_message(digest)
        log.info("Push %s: sent digest (%d chars, %d tickers analyzed).", push.name, len(digest), len(results))
    except TelegramError as e:
        log.error("Telegram send failed for %s: %s", push.name, e)
    except Exception as e:
        log.exception("Push job %s crashed: %s", push.name, e)


def _parse_cron(expr: str) -> CronTrigger:
    """Parse a 5-field crontab expression: minute hour day month day_of_week."""
    parts = expr.split()
    if len(parts) != 5:
        raise ValueError(f"Cron expression must have 5 fields, got: {expr!r}")
    minute, hour, day, month, dow = parts
    return CronTrigger(
        minute=minute, hour=hour, day=day, month=month, day_of_week=dow,
    )


def run() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    sched_cfg = load_schedule()
    log.info("Loading scheduler with timezone=%s, %d pushes", sched_cfg.host_timezone, len(sched_cfg.pushes))

    scheduler = BlockingScheduler(timezone=sched_cfg.host_timezone)
    for push in sched_cfg.pushes:
        trigger = _parse_cron(push.cron)
        scheduler.add_job(
            _job,
            trigger=trigger,
            args=[push],
            id=push.name,
            name=push.name,
            misfire_grace_time=600,  # 10 min grace
            coalesce=True,            # if missed multiple, run once
            max_instances=1,
        )
        log.info("Registered push %r with cron %r", push.name, push.cron)

    def shutdown(signum, frame):
        log.info("Signal %d received — shutting down scheduler.", signum)
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)

    log.info("Scheduler started. Press Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    run()
