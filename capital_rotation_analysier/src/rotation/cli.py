from __future__ import annotations

import logging
import os
import uuid
from datetime import date, datetime, timedelta
from pathlib import Path

import click
import pandas as pd
import pandas_market_calendars as mcal

from .alerts import dispatch, evaluate_triggers
from .compute import backfill_signals, run_signals_for_date
from .config import load_config
from .ingest.flows_adapter import snapshot_today as snapshot_flows_today
from .ingest.fred_adapter import fetch_and_store_all as fetch_fred_all
from .ingest.runner import fetch_validate_store
from .pipeline import run_daily
from .validate import latest_verdicts, run_validation
from .regime import run_regime_for_date
from .report import build_daily_report, build_monthly_report, build_weekly_report, store_report
from .report_hk import build_hk_daily_report
from .store import connect, load_signals_and_regime

log = logging.getLogger("rotate")


def _parse_date(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def _us_trading_days(start: date, end: date) -> list[date]:
    cal = mcal.get_calendar("NYSE")
    sched = cal.schedule(start_date=start.isoformat(), end_date=end.isoformat())
    return [d.date() for d in pd.to_datetime(sched.index)]


def _run_for_date(cfg, asof: date) -> dict:
    run_id = f"rotate-{asof.isoformat()}-{uuid.uuid4().hex[:8]}"
    log.info("run %s asof=%s symbols=%d", run_id, asof, len(cfg.universe))
    with connect(cfg.storage.duckdb_path) as con:
        return fetch_validate_store(cfg, con, run_id, asof)


# Secrets can come from either env vars (local dev with .env) or CLI flags
# (cloud deploy where there's no on-disk .env). Each flag, if provided, sets
# the corresponding env var BEFORE any subcommand runs so the existing
# os.environ lookups in alerts.py / llm_interpret.py / fred_adapter.py work
# transparently. CLI flag wins over env var.
_KEY_FLAG_TO_ENV = {
    "telegram_token":    "TELEGRAM_BOT_TOKEN",
    "telegram_chat_id":  "TELEGRAM_CHAT_ID",
    "deepseek_key":      "DEEPSEEK_API_KEY",
    "fred_key":          "FRED_API_KEY",
}


@click.group()
@click.option("--config", "config_path", default="config.yaml", show_default=True)
@click.option("-v", "--verbose", is_flag=True)
@click.option("--telegram-token", "telegram_token", default=None,
              help="Telegram bot token. Cloud deploy: pass this instead of TELEGRAM_BOT_TOKEN env var.")
@click.option("--telegram-chat-id", "telegram_chat_id", default=None,
              help="Telegram chat ID. Cloud deploy: pass this instead of TELEGRAM_CHAT_ID env var.")
@click.option("--deepseek-key", "deepseek_key", default=None,
              help="DeepSeek API key for LLM Section 12. Cloud deploy: pass this instead of DEEPSEEK_API_KEY env var.")
@click.option("--fred-key", "fred_key", default=None,
              help="FRED API key for RRP / curve series. Cloud deploy: pass this instead of FRED_API_KEY env var.")
@click.pass_context
def cli(
    ctx: click.Context,
    config_path: str,
    verbose: bool,
    telegram_token: str | None,
    telegram_chat_id: str | None,
    deepseek_key: str | None,
    fred_key: str | None,
):
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    # Promote any provided flag to env var so downstream code sees it.
    flag_values = {
        "telegram_token": telegram_token,
        "telegram_chat_id": telegram_chat_id,
        "deepseek_key": deepseek_key,
        "fred_key": fred_key,
    }
    for flag, value in flag_values.items():
        if value:
            os.environ[_KEY_FLAG_TO_ENV[flag]] = value

    ctx.ensure_object(dict)
    ctx.obj["cfg"] = load_config(config_path)


@cli.command("fetch")
@click.option("--date", "asof", type=str, default=None,
              help="YYYY-MM-DD. Defaults to most recent NYSE trading day on or before today.")
@click.option("--range", "date_range", type=str, default=None,
              help="YYYY-MM-DD..YYYY-MM-DD inclusive. Fetches each NYSE trading day in range.")
@click.pass_context
def fetch_cmd(ctx: click.Context, asof: str | None, date_range: str | None):
    """Fetch and store bars for one date or a range."""
    cfg = ctx.obj["cfg"]

    if date_range:
        a, b = date_range.split("..")
        days = _us_trading_days(_parse_date(a), _parse_date(b))
        if not days:
            click.echo(f"No trading days in range {date_range}", err=True)
            return
    else:
        if asof:
            target = _parse_date(asof)
        else:
            target = date.today()
        # Snap to most recent trading day <= target (handles weekends/holidays)
        cal_days = _us_trading_days(target - timedelta(days=10), target)
        if not cal_days:
            click.echo(f"No trading day at/before {target}", err=True)
            return
        days = [cal_days[-1]]

    for d in days:
        out = _run_for_date(cfg, d)
        click.echo(
            f"[{out['asof']}] fetched={out['fetched']} inserted={out['inserted']} "
            f"updated={out['updated']} quarantined={out['quarantined']} "
            f"coverage={out['coverage']:.0%}{'' if out['coverage_ok'] else ' DEGRADED'} "
            f"run_id={out['run_id']}"
        )


@cli.command("signals")
@click.option("--date", "asof", type=str, default=None,
              help="YYYY-MM-DD. Defaults to most recent NYSE trading day.")
@click.option("--range", "date_range", type=str, default=None,
              help="YYYY-MM-DD..YYYY-MM-DD inclusive. Computes signals for each trading day.")
@click.pass_context
def signals_cmd(ctx: click.Context, asof: str | None, date_range: str | None):
    """Compute and store metrics + 8 signal scores for the given date(s)."""
    cfg = ctx.obj["cfg"]

    if date_range:
        a, b = date_range.split("..")
        # Use the batch helper: loads the bar panel once and iterates internally.
        # Per-day run_signals_for_date re-loads the panel each call (~50ms × 2500
        # days = 125s of redundant queries); batch is ~1-2 seconds for the panel
        # load plus ~5ms per asof.
        out = backfill_signals(cfg, _parse_date(a), _parse_date(b))
        if out.get("skipped"):
            click.echo(f"[{out['start']}..{out['end']}] SKIPPED: {out['skipped']}")
            return
        click.echo(
            f"[{out['start']}..{out['end']}] backfilled days={out['n_days']} "
            f"metric_rows={out['n_metric_rows']} signal_rows={out['n_signal_rows']} "
            f"symbols={out['n_symbols']}"
        )
        return

    target = _parse_date(asof) if asof else date.today()
    cal_days = _us_trading_days(target - timedelta(days=10), target)
    if not cal_days:
        click.echo(f"No trading day at/before {target}", err=True)
        return
    days = [cal_days[-1]]

    for d in days:
        out = run_signals_for_date(cfg, d)
        if out.get("skipped"):
            click.echo(f"[{out['asof']}] SKIPPED: {out['skipped']}")
            continue
        scores = out["signals_summary"]
        parts = []
        for k, v in scores.items():
            s = v["score"]; c = v["confidence"]
            s_str = "-" if s is None else f"{s:+.1f}"
            c_str = "-" if c is None else f"{c:.2f}"
            parts.append(f"{k}={s_str}@{c_str}")
        click.echo(
            f"[{out['asof']}] n_metrics={out['n_metric_rows']} "
            f"n_signals={out['n_signals']}  " + "  ".join(parts)
        )


@cli.command("flows")
@click.option("--date", "asof", type=str, default=None)
@click.pass_context
def flows_cmd(ctx: click.Context, asof: str | None):
    """Snapshot today's ETF shares-outstanding (Method B, universal AUM-delta proxy)."""
    cfg = ctx.obj["cfg"]
    target = _parse_date(asof) if asof else date.today()
    cal_days = _us_trading_days(target - timedelta(days=10), target)
    if not cal_days:
        click.echo(f"No trading day at/before {target}", err=True); return
    d = cal_days[-1]
    with connect(cfg.storage.duckdb_path) as con:
        out = snapshot_flows_today(cfg, con, d)
    click.echo(
        f"[{out['asof']}] issuer_direct={out.get('n_issuer_direct', 0)}  "
        f"method_b={out.get('n_method_b', 0)}  "
        f"deltas={out.get('n_with_delta', 0)}  "
        f"upserted={out.get('n_upserted', 0)}"
    )
    if out.get("issuer_failures"):
        click.echo(f"  Issuer-direct failed for: {', '.join(out['issuer_failures'])} (falling back to Method B if available)")
    click.echo(
        f"  Priority issuer-direct picks: {', '.join(out.get('priority_issuer_direct_candidates', []))}"
    )


@cli.command("fred")
@click.option("--start", "start", type=str, default="2024-11-01")
@click.pass_context
def fred_cmd(ctx: click.Context, start: str):
    """Refresh FRED macro series (RRPONTSYD, T10Y2Y). Requires FRED_API_KEY env var."""
    cfg = ctx.obj["cfg"]
    with connect(cfg.storage.duckdb_path) as con:
        results = fetch_fred_all(con, start=_parse_date(start))
    if not results:
        click.echo("FRED_API_KEY not set; nothing fetched. See .env.example.")
        return
    for sid, n in results.items():
        click.echo(f"  {sid}: {n} rows upserted")


@cli.command("validate")
@click.option("--date", "asof", type=str, default=None)
@click.pass_context
def validate_cmd(ctx: click.Context, asof: str | None):
    """Run the IC + hit-rate harness on all 8 signals. Persists verdicts."""
    cfg = ctx.obj["cfg"]
    target = _parse_date(asof) if asof else date.today()
    verdicts = run_validation(cfg, target)
    if not verdicts:
        click.echo("No verdicts produced (no bar history?)"); return
    click.echo(f"Validation @ {target.isoformat()}:")
    for v in verdicts:
        ic = f"IC5d={v.median_ic_5d:+.3f}" if v.median_ic_5d is not None else "IC5d=—"
        pct = f"pos={v.pct_windows_pos_ic:.2f}" if v.pct_windows_pos_ic is not None else "pos=—"
        hr = f"hit={v.hit_rate_overall:.2f}" if v.hit_rate_overall is not None else "hit=—"
        click.echo(
            f"  [{v.verdict:13s}] {v.signal_name:20s}  "
            f"{ic}  {pct}  {hr}  n={v.n_observations}  fwd={v.forward_asset or '—'}"
        )
        if v.verdict != "pass":
            click.echo(f"      reason: {v.reason}")


@cli.command("scorecard")
@click.option("--date", "asof", type=str, default=None,
              help="YYYY-MM-DD. Defaults to most recent NYSE trading day.")
@click.option("--backfill", "backfill_range", type=str, default=None,
              help="YYYY-MM-DD..YYYY-MM-DD: record forecasts for every signal day "
                   "in range (no lookahead — each day only uses prior analogues), "
                   "then resolve everything whose horizon has elapsed.")
@click.pass_context
def scorecard_cmd(ctx: click.Context, asof: str | None, backfill_range: str | None):
    """Record/resolve published forecasts (W1) and print the hit-rate scorecard."""
    from .scorecard import (
        FORECAST_TYPES, backfill_scorecard, load_scorecard,
        record_forecasts, resolve_forecasts,
    )
    cfg = ctx.obj["cfg"]

    if backfill_range:
        a, b = backfill_range.split("..")
        out = backfill_scorecard(cfg, _parse_date(a), _parse_date(b))
        click.echo(
            f"[{out['start']}..{out['end']}] days={out['days']} "
            f"recorded={out['recorded']} resolved={out['resolved']}"
        )
        target = _parse_date(b)
    else:
        target = _parse_date(asof) if asof else date.today()
        cal_days = _us_trading_days(target - timedelta(days=10), target)
        if not cal_days:
            click.echo(f"No trading day at/before {target}", err=True)
            return
        target = cal_days[-1]
        rec = record_forecasts(cfg, target)
        res = resolve_forecasts(cfg, target)
        click.echo(f"[{target}] recorded={rec['recorded']} resolved={res['resolved']}")

    with connect(cfg.storage.duckdb_path) as con:
        sc = load_scorecard(con, target)
    click.echo(f"Scorecard @ {target.isoformat()} (last 100 resolved per type):")
    for ftype in FORECAST_TYPES:
        s = sc["summary"][ftype]
        hr = f"{s['hit_rate']*100:.0f}%" if s["hit_rate"] is not None else "—"
        click.echo(
            f"  {ftype:12s} hit_rate={hr:>4s}  resolved={s['n_resolved']:3d}  "
            f"pending={s['n_pending']}"
        )


@cli.command("daily")
@click.option("--date", "asof", type=str, default=None,
              help="YYYY-MM-DD. Defaults to today (snapped to last NYSE trading day).")
@click.pass_context
def daily_cmd(ctx: click.Context, asof: str | None):
    """Run the full daily pipeline: fetch → signals → regime → report → alerts → heartbeat.

    Suitable as a cloud cron entry. Heartbeat to Telegram confirms success / flags failure.
    """
    cfg = ctx.obj["cfg"]
    target = _parse_date(asof) if asof else date.today()
    summary = run_daily(cfg, target)
    if summary.get("ok"):
        click.echo(f"OK  asof={summary['asof']}  steps={len(summary['steps'])}")
        for s in summary["steps"]:
            click.echo(f"  ✓ {s['step']} [{s['duration_s']:.1f}s]")
    else:
        click.echo(f"FAIL asof={summary.get('asof')}")
        for s in summary.get("steps", []):
            mark = "✓" if s["ok"] else "✗"
            click.echo(f"  {mark} {s['step']} [{s['duration_s']:.1f}s]"
                       + ("" if s["ok"] else f"  {s.get('error','')}"))
    if summary.get("heartbeat"):
        click.echo(f"heartbeat: {summary['heartbeat']}")
    for key in ("pdf_attachment_us", "pdf_attachment_hk"):
        if summary.get(key):
            click.echo(f"{key}: {summary[key]}")


@cli.command("heartbeat")
@click.pass_context
def heartbeat_cmd(ctx: click.Context):
    """Send a heartbeat alert without running the full pipeline (sanity check Telegram wiring)."""
    cfg = ctx.obj["cfg"]
    from .alerts import Alert
    a = Alert(
        alert_type="heartbeat_manual", priority="P3",
        headline=f"Manual heartbeat — {date.today().isoformat()}",
        body="Manual `rotate heartbeat` invocation. If you see this, Telegram wiring works.",
        ts=date.today(),
    )
    out = dispatch(cfg, [a])
    click.echo(out)


@cli.command("regime")
@click.option("--date", "asof", type=str, default=None)
@click.option("--range", "date_range", type=str, default=None)
@click.option("--horizon", type=click.Choice(["daily", "weekly", "monthly"]), default="daily")
@click.pass_context
def regime_cmd(ctx: click.Context, asof: str | None, date_range: str | None, horizon: str):
    """Classify the market regime for the date(s)."""
    cfg = ctx.obj["cfg"]
    if date_range:
        a, b = date_range.split("..")
        days = _us_trading_days(_parse_date(a), _parse_date(b))
    else:
        target = _parse_date(asof) if asof else date.today()
        cal_days = _us_trading_days(target - timedelta(days=10), target)
        if not cal_days:
            click.echo(f"No trading day at/before {target}", err=True); return
        days = [cal_days[-1]]
    for d in days:
        r = run_regime_for_date(cfg, d, horizon=horizon)
        if r.get("skipped"):
            click.echo(f"[{d}] SKIPPED: {r['skipped']}"); continue
        flip = " (FLIP)" if r["flipped"] else ""
        click.echo(
            f"[{r['asof']}] regime={r['regime']}  proposed={r['proposed']}  "
            f"days_in={r['days_in_regime']}  conf={r['confidence']:.2f}{flip}"
        )


@cli.command("alerts")
@click.option("--date", "asof", type=str, default=None)
@click.pass_context
def alerts_cmd(ctx: click.Context, asof: str | None):
    """Evaluate alert triggers for date and dispatch through configured channels."""
    cfg = ctx.obj["cfg"]
    target = _parse_date(asof) if asof else date.today()
    cal_days = _us_trading_days(target - timedelta(days=10), target)
    if not cal_days:
        click.echo(f"No trading day at/before {target}", err=True); return
    d = cal_days[-1]

    with connect(cfg.storage.duckdb_path) as con:
        signals, today_regime, prev_regime = load_signals_and_regime(con, d)

    alerts = evaluate_triggers(signals, today_regime, prev_regime, d)
    if not alerts:
        click.echo(f"[{d}] No alerts triggered.")
        return
    results = dispatch(cfg, alerts)
    for r in results:
        click.echo(f"[{d}] {r}")


@cli.command("report")
@click.option("--date", "asof", type=str, default=None,
              help="YYYY-MM-DD. Defaults to most recent trading day in store.")
@click.option("--horizon", type=click.Choice(["daily", "weekly", "monthly"]), default="daily")
@click.option("--market", type=click.Choice(["us", "hk"]), default="us",
              help="us = global/NYSE-panel report; hk = independent Hong Kong report (daily only).")
@click.option("--out", "out_path", type=click.Path(), default=None,
              help="Write to file as well as printing. Default writes to reports/<asof>_<horizon>.md")
@click.option("--pdf/--no-pdf", default=False, help="Also render a PDF next to the markdown.")
@click.pass_context
def report_cmd(ctx: click.Context, asof: str | None, horizon: str, market: str,
               out_path: str | None, pdf: bool):
    """Build a markdown report from stored signals + metrics, optionally also as PDF."""
    cfg = ctx.obj["cfg"]
    if asof:
        target = _parse_date(asof)
    else:
        target = date.today()
    cal_days = _us_trading_days(target - timedelta(days=10), target)
    if not cal_days:
        click.echo(f"No trading day at/before {target}", err=True)
        return
    d = cal_days[-1]

    if market == "hk":
        if horizon != "daily":
            click.echo("The HK report only has a daily horizon.", err=True)
            return
        body = build_hk_daily_report(cfg, d)
        horizon = "daily_hk"
    else:
        builder = {"daily": build_daily_report, "weekly": build_weekly_report, "monthly": build_monthly_report}[horizon]
        body = builder(cfg, d)
    store_report(cfg, d, horizon, body)

    from pathlib import Path
    out = out_path or f"reports/{d.isoformat()}_{horizon}.md"
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(body, encoding="utf-8")
    click.echo(f"Wrote {out} ({len(body)} chars)")

    if pdf:
        from .pdf_report import markdown_to_pdf
        pdf_path = Path(out).with_suffix(".pdf")
        markdown_to_pdf(body, pdf_path)
        click.echo(f"Wrote {pdf_path} ({pdf_path.stat().st_size:,} bytes)")


@cli.command("status")
@click.pass_context
def status_cmd(ctx: click.Context):
    """Show row counts and the last few runs."""
    cfg = ctx.obj["cfg"]
    with connect(cfg.storage.duckdb_path) as con:
        n_bars = con.execute("SELECT COUNT(*) FROM raw_bars").fetchone()[0]
        n_q = con.execute("SELECT COUNT(*) FROM raw_bars_quarantine").fetchone()[0]
        dist = con.execute(
            "SELECT symbol, COUNT(*) AS n, MIN(ts) AS first_ts, MAX(ts) AS last_ts "
            "FROM raw_bars GROUP BY symbol ORDER BY symbol"
        ).df()
        recent = con.execute(
            "SELECT run_id, asof_date, status, n_inserted, n_updated, n_quarantined "
            "FROM run_log ORDER BY started_at DESC LIMIT 10"
        ).df()

    click.echo(f"raw_bars rows:        {n_bars}")
    click.echo(f"quarantined rows:     {n_q}")
    click.echo("")
    click.echo("Coverage per symbol:")
    click.echo(dist.to_string(index=False))
    click.echo("")
    click.echo("Recent runs:")
    click.echo(recent.to_string(index=False))


if __name__ == "__main__":
    cli()
