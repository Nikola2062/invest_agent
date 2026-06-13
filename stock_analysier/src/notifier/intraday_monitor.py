"""Intraday Monitor — lightweight, no-LLM, runs every 15 min during market hours.

For each held position:
  1. Fetch current intraday price (yfinance fast)
  2. Look up the latest audit row → gets rebuy band + intrinsic value + last-known levels
  3. Check intraday triggers:
       - rebuy_band_entry: price entered a pre-committed rebuy band → ALERT (BUY opportunity)
       - intraday_drop:    price down ≥ 5% from prior close → ALERT
       - intraday_spike:   price up ≥ 8% from prior close → ALERT (consider taking profit)
       - intrinsic_cross:  price crossed BELOW intrinsic_low or ABOVE intrinsic_high → ALERT
  4. Send Telegram if any fired, mark alerts as pushed

Designed to be cheap: a few yfinance calls + DB read. No LLM in the loop.

Run with:
    .venv/bin/python -m src.notifier.intraday_monitor      # one-shot
    ./run_intraday.sh                                       # daemon (15-min cron)
"""
from __future__ import annotations

import json
import logging
import signal
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import yfinance as yf
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src.config.loader import load_portfolio, load_risk_policy, load_schedule
from src.notifier.telegram_client import TelegramError, send_message
from src.storage.audit import (
    get_latest_run,
    get_unpushed_alerts,
    mark_alert_pushed,
    record_alert,
)

log = logging.getLogger(__name__)


@dataclass
class IntradayQuote:
    symbol: str
    current_price: float
    prior_close: float
    session_open: Optional[float]
    pct_from_prior_close: float


def _quote(symbol: str) -> Optional[IntradayQuote]:
    try:
        t = yf.Ticker(symbol)
        h = t.history(period="5d", interval="1d", auto_adjust=False)["Close"].dropna()
        if len(h) < 2:
            return None
        prior = float(h.iloc[-2])
        intraday = t.history(period="1d", interval="5m")["Close"].dropna()
        if intraday.empty:
            current = float(h.iloc[-1])
            session_open = current
        else:
            current = float(intraday.iloc[-1])
            session_open = float(intraday.iloc[0])
        pct = (current / prior - 1.0) * 100.0
        return IntradayQuote(
            symbol=symbol, current_price=current, prior_close=prior,
            session_open=session_open, pct_from_prior_close=pct,
        )
    except Exception as e:
        log.warning("Intraday quote failed for %s: %s", symbol, e)
        return None


def _extract_audit_levels(row) -> dict:
    """Pull rebuy band + intrinsic levels from the latest audit row."""
    levels = {
        "rebuy_band_low": None,
        "rebuy_band_high": None,
        "intrinsic_low": row["intrinsic_low"],
        "intrinsic_high": row["intrinsic_high"],
        "intrinsic_base": row["intrinsic_base"],
    }
    try:
        full = json.loads(row["full_result_json"])
        held = (full.get("if_held") or {}).get("tactical") or {}
        levels["rebuy_band_low"] = held.get("rebuy_band_low")
        levels["rebuy_band_high"] = held.get("rebuy_band_high")
    except Exception:
        pass
    return levels


def check_intraday() -> list[dict]:
    """One-shot check across all held positions. Returns alerts created."""
    portfolio = load_portfolio()
    thresholds = load_risk_policy().intraday_thresholds
    created: list[dict] = []
    for h in portfolio.holdings:
        q = _quote(h.symbol)
        if q is None:
            continue
        latest = get_latest_run(h.symbol)
        if latest is None:
            log.info("No prior audit for %s — skipping intraday check.", h.symbol)
            continue
        levels = _extract_audit_levels(latest)

        # 1. Intraday drop / spike
        if q.pct_from_prior_close <= thresholds.drop_pct:
            evidence = {"pct_from_prior_close": q.pct_from_prior_close,
                        "current_price": q.current_price, "prior_close": q.prior_close}
            aid = record_alert(
                h.symbol, "warning", "intraday_drop",
                f"{h.symbol} down {q.pct_from_prior_close:+.1f}% intraday "
                f"({q.prior_close:.2f} → {q.current_price:.2f})",
                evidence,
            )
            created.append({"id": aid, "category": "intraday_drop"})
        if q.pct_from_prior_close >= thresholds.spike_pct:
            evidence = {"pct_from_prior_close": q.pct_from_prior_close,
                        "current_price": q.current_price, "prior_close": q.prior_close}
            aid = record_alert(
                h.symbol, "info", "intraday_spike",
                f"{h.symbol} up {q.pct_from_prior_close:+.1f}% intraday — consider profit-taking",
                evidence,
            )
            created.append({"id": aid, "category": "intraday_spike"})

        # 2. Rebuy band entry — only if a rebuy band was previously recorded
        rb_lo = levels.get("rebuy_band_low")
        rb_hi = levels.get("rebuy_band_high")
        if rb_lo is not None and rb_hi is not None and rb_lo <= q.current_price <= rb_hi:
            aid = record_alert(
                h.symbol, "warning", "rebuy_band_entry",
                f"{h.symbol} entered pre-committed rebuy band {rb_lo:.2f}–{rb_hi:.2f} (now {q.current_price:.2f})",
                {"current_price": q.current_price, "rebuy_band_low": rb_lo, "rebuy_band_high": rb_hi},
            )
            created.append({"id": aid, "category": "rebuy_band_entry"})

        # 3. Intrinsic value crosses
        if levels["intrinsic_low"] and q.current_price < levels["intrinsic_low"]:
            # Cross from above → below
            aid = record_alert(
                h.symbol, "info", "intrinsic_low_cross",
                f"{h.symbol} now below intrinsic_low ({q.current_price:.2f} < {levels['intrinsic_low']:.2f}) — value zone",
                {"current_price": q.current_price, "intrinsic_low": levels["intrinsic_low"]},
            )
            created.append({"id": aid, "category": "intrinsic_low_cross"})

    return created


def push_unpushed() -> int:
    """Send any unpushed alerts to Telegram, mark them pushed."""
    unpushed = get_unpushed_alerts()
    if not unpushed:
        return 0
    from src.notifier.formatter import format_thesis_break_alerts
    msg = format_thesis_break_alerts(unpushed)
    msg = "*⏱ Intraday alerts*\n\n" + msg
    try:
        send_message(msg)
        for row in unpushed:
            mark_alert_pushed(row["id"])
        return len(unpushed)
    except TelegramError as e:
        log.error("Intraday alert push failed: %s", e)
        return 0


def run_once() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    log.info("Intraday Monitor: one-shot check…")
    created = check_intraday()
    log.info("Intraday Monitor: created %d new alerts.", len(created))
    pushed = push_unpushed()
    if pushed:
        log.info("Pushed %d alerts to Telegram.", pushed)


def run_daemon(cron_expr: str = "*/15 14-22 * * 1-5") -> None:
    """Run as a long-running daemon. Default: every 15 min between 14:00-22:00 Berlin (US session)."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    sched_cfg = load_schedule()
    log.info(
        "Intraday Monitor daemon. Timezone=%s, cron=%r", sched_cfg.host_timezone, cron_expr,
    )

    scheduler = BlockingScheduler(timezone=sched_cfg.host_timezone)
    parts = cron_expr.split()
    if len(parts) != 5:
        raise ValueError(f"Cron expression must have 5 fields, got: {cron_expr!r}")
    minute, hour, day, month, dow = parts
    trigger = CronTrigger(minute=minute, hour=hour, day=day, month=month, day_of_week=dow)

    def _tick():
        try:
            created = check_intraday()
            if created:
                log.info("Intraday tick: %d new alerts.", len(created))
            push_unpushed()
        except Exception as e:
            log.exception("Intraday tick failed: %s", e)

    scheduler.add_job(_tick, trigger=trigger, id="intraday", coalesce=True, max_instances=1)

    def shutdown(signum, frame):
        log.info("Signal %d — shutting down intraday monitor.", signum)
        scheduler.shutdown(wait=False)
        sys.exit(0)

    signal.signal(signal.SIGTERM, shutdown)
    signal.signal(signal.SIGINT, shutdown)
    log.info("Intraday Monitor started. Ctrl+C to stop.")
    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        pass


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Intraday Monitor")
    parser.add_argument("--daemon", action="store_true", help="Run as a daemon on a 15-min cron")
    parser.add_argument("--cron", default="*/15 14-22 * * 1-5", help="Crontab expression for daemon mode")
    args = parser.parse_args()
    if args.daemon:
        run_daemon(args.cron)
    else:
        run_once()
