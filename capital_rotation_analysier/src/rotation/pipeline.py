"""Daily orchestration + heartbeat per cloud-host requirement.

`run_daily(cfg)` runs the full pipeline for the most recent NYSE trading day:
fetch → signals → regime → report → alerts. At the end it sends a heartbeat
Alert (priority P3) to Telegram so the operator sees the system is healthy.

If any step fails, the heartbeat carries the failure detail and is bumped to P1.
"""
from __future__ import annotations

import logging
import os
import traceback
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pandas_market_calendars as mcal

from .alerts import (
    Alert, dispatch, evaluate_triggers, telegram_send_document,
    validation_failure_alerts,
)
from .compute import run_signals_for_date
from .config import Config
from .ingest.flows_adapter import snapshot_today as snapshot_flows_today
from .ingest.fred_adapter import fetch_and_store_all as fetch_fred_all
from .ingest.china_macro_adapter import (
    fetch_and_store_all as fetch_china_macro_all,
)
from .ingest.stock_connect_adapter import (
    fetch_and_store_all as fetch_stock_connect_all,
    refresh_holdings as refresh_sb_holdings,
)
from .pdf_report import markdown_to_pdf
from .regime import run_regime_for_date
from .report import build_daily_report, store_report
from .report_hk import build_hk_daily_report
from .store import connect, load_signals_and_regime
from .validate import latest_verdicts, run_validation

log = logging.getLogger(__name__)


def _last_trading_day(target: date) -> date | None:
    cal = mcal.get_calendar("NYSE")
    sched = cal.schedule(
        start_date=(target - timedelta(days=10)).isoformat(),
        end_date=target.isoformat(),
    )
    if sched.empty:
        return None
    return pd.to_datetime(sched.index[-1]).date()


def _step(name: str, fn) -> dict:
    """Run a pipeline step, capturing success/failure as structured data."""
    t0 = datetime.utcnow()
    try:
        out = fn()
        return {"step": name, "ok": True, "duration_s": (datetime.utcnow() - t0).total_seconds(),
                "result": out}
    except Exception as exc:
        return {"step": name, "ok": False, "duration_s": (datetime.utcnow() - t0).total_seconds(),
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()}


def run_daily(cfg: Config, target: date | None = None) -> dict:
    """Execute the full daily pipeline. Always returns a structured summary;
    never raises (errors are captured per-step so the heartbeat still fires)."""
    from .ingest.runner import fetch_validate_store
    import uuid

    target = target or date.today()
    asof = _last_trading_day(target)
    if asof is None:
        return {"asof": None, "ok": False, "error": "no_trading_day_at_or_before_target"}

    steps: list[dict] = []

    # 1. Fetch bars (shared runner; logs the run degraded below the §1.7 gate)
    def _do_fetch():
        run_id = f"daily-{asof.isoformat()}-{uuid.uuid4().hex[:6]}"
        with connect(cfg.storage.duckdb_path) as con:
            return fetch_validate_store(cfg, con, run_id, asof)
    fetch_step = _step("fetch", _do_fetch)
    steps.append(fetch_step)
    fetch_res = fetch_step.get("result") or {}
    coverage_ok = bool(fetch_step["ok"]) and bool(fetch_res.get("coverage_ok", True))

    # 1b. FRED macro series (only if FRED_API_KEY is set; otherwise no-op).
    def _do_fred():
        with connect(cfg.storage.duckdb_path) as con:
            return fetch_fred_all(con, end=asof)  # start defaults to FRED_START_DATE
    steps.append(_step("fred", _do_fred))

    # 1c. ETF flows snapshot (Method B — universal AUM-delta proxy).
    def _do_flows():
        with connect(cfg.storage.duckdb_path) as con:
            return snapshot_flows_today(cfg, con, asof)
    steps.append(_step("flows", _do_flows))

    # 1d. Stock Connect Southbound aggregate flows (akshare/Eastmoney, D2).
    # No API key needed. Non-blocking — if akshare or the endpoint fails the
    # rest of the pipeline must still run.
    def _do_stock_connect():
        with connect(cfg.storage.duckdb_path) as con:
            agg = fetch_stock_connect_all(con)
            # D3: per-stock Southbound holdings (7-day rolling refresh, ~14 HTTP
            # calls — kept small so a single bad endpoint day doesn't hold up
            # the rest of the pipeline).
            holdings = refresh_sb_holdings(con, asof, lookback_days=7)
            return {**agg, "holdings_rows": holdings}
    steps.append(_step("stock_connect", _do_stock_connect))

    # 1e. China macro — PBOC RRR/LPR + SHIBOR via akshare/Eastmoney (D4).
    # All series refresh fully on each run; events series (RRR, LPR) only
    # publish when PBOC moves, so daily upserts are near-zero work.
    def _do_china_macro():
        with connect(cfg.storage.duckdb_path) as con:
            return fetch_china_macro_all(con)
    steps.append(_step("china_macro", _do_china_macro))

    # 2. Signals — gated on §1.7 coverage: never silently computed on a panel
    # that is missing a large slice of the universe.
    def _do_signals():
        if not coverage_ok:
            raise RuntimeError(
                "insufficient_data: bar coverage "
                f"{fetch_res.get('coverage', 0):.0%} of NYSE-aligned universe is below the "
                f"{cfg.ingest.min_coverage_pct:.0f}% gate — signals skipped (design §1.7)"
            )
        return run_signals_for_date(cfg, asof)
    steps.append(_step("signals", _do_signals))

    # 3. Regime
    steps.append(_step("regime", lambda: run_regime_for_date(cfg, asof, "daily")))

    # 4. Validation harness (rolling-IC + hit-rate per §3.7). Capture prior
    # verdicts BEFORE running so we can detect new failures and notify.
    # Validation must run BEFORE the report so the report's Detected Themes
    # table can show the live verdict tags.
    prior_verdicts = latest_verdicts(cfg)
    def _do_validate():
        return run_validation(cfg, asof)
    steps.append(_step("validate", _do_validate))
    new_verdicts = latest_verdicts(cfg)

    # 4b. Forecast scorecard (W1): record today's published outlooks and
    # resolve any whose horizon has elapsed. Runs BEFORE the report so the
    # Forecast Scorecard section reflects today's state.
    def _do_scorecard():
        from .scorecard import record_forecasts, resolve_forecasts
        rec = record_forecasts(cfg, asof, verdicts=new_verdicts)
        res = resolve_forecasts(cfg, asof)
        return {"recorded": rec["recorded"], "resolved": res["resolved"]}
    steps.append(_step("scorecard", _do_scorecard))

    # 5. Reports — US/global and Hong Kong are independent documents (separate
    # universes, separate trading calendars). Each renders markdown + PDF; both
    # PDFs get attached to the Telegram heartbeat.
    report_paths: dict[str, Path] = {}

    def _render(body: str, md_path: Path, key: str) -> dict:
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text(body, encoding="utf-8")
        report_paths[f"md_{key}"] = md_path
        pdf_path = md_path.with_suffix(".pdf")
        try:
            markdown_to_pdf(body, pdf_path)
            report_paths[f"pdf_{key}"] = pdf_path
        except Exception as exc:
            log.warning("PDF render failed (%s): %s; markdown still written", key, exc)
        return {"md": str(md_path), "bytes": len(body),
                "pdf": str(pdf_path) if f"pdf_{key}" in report_paths else None}

    def _do_report():
        body = build_daily_report(cfg, asof)
        store_report(cfg, asof, "daily", body)
        return _render(body, Path(f"reports/{asof.isoformat()}_daily.md"), "us")
    steps.append(_step("report", _do_report))

    def _do_report_hk():
        body = build_hk_daily_report(cfg, asof)
        store_report(cfg, asof, "daily_hk", body)
        return _render(body, Path(f"reports/{asof.isoformat()}_daily_hk.md"), "hk")
    steps.append(_step("report_hk", _do_report_hk))

    # 5. Alerts (regime change, strong rotations, validation failures, etc.)
    def _do_alerts():
        with connect(cfg.storage.duckdb_path) as con:
            signals, today_regime, prev_regime = load_signals_and_regime(con, asof)

        alerts = evaluate_triggers(signals, today_regime, prev_regime, asof, verdicts=new_verdicts)
        alerts += validation_failure_alerts(new_verdicts, prior_verdicts, asof)
        return dispatch(cfg, alerts) if alerts else {"alerts_fired": 0}
    steps.append(_step("alerts", _do_alerts))

    # 6. Heartbeat — always last, always fires
    summary = {
        "asof": asof.isoformat(),
        "steps": steps,
        "ok": all(s["ok"] for s in steps),
    }

    failed = [s for s in steps if not s["ok"]]
    if failed:
        headline = f"Daily pipeline FAILED ({len(failed)}/{len(steps)} steps) — {asof.isoformat()}"
        body_lines = [
            f"Failed steps: {', '.join(s['step'] for s in failed)}",
            "",
            "Details:",
        ]
        for s in failed:
            body_lines.append(f"  • {s['step']}: {s.get('error', 'unknown')}")
        priority = "P1"
    else:
        headline = f"Daily pipeline OK — {asof.isoformat()}"
        body_lines = [f"All {len(steps)} steps completed. Trading day: {asof.isoformat()}", ""]
        for s in steps:
            r = s.get("result", {})
            tag = ""
            if s["step"] == "fetch" and isinstance(r, dict):
                tag = f"  ({r.get('inserted', 0)} new bars, {r.get('quarantined', 0)} quarantined)"
            elif s["step"] == "signals" and isinstance(r, dict):
                tag = f"  ({r.get('n_signals', 0)} signals)"
            elif s["step"] == "regime" and isinstance(r, dict):
                tag = f"  ({r.get('regime', '—')}{' FLIP' if r.get('flipped') else ''})"
            elif s["step"] == "alerts":
                n = len(r) if isinstance(r, list) else 0
                tag = f"  ({n} alerts fired)"
            body_lines.append(f"  ✓ {s['step']}{tag}  [{s['duration_s']:.1f}s]")
        priority = "P3"

    heartbeat = Alert(
        alert_type="heartbeat",
        priority=priority,
        headline=headline,
        body="\n".join(body_lines),
        ts=asof,
    )
    summary["heartbeat"] = dispatch(cfg, [heartbeat])

    # Attach the PDF reports to Telegram as follow-up documents, so the user
    # gets the heartbeat text + both full reports in chat.
    status_tag = "OK" if not failed else "FAILED"
    telegram_results: dict[str, dict] = {}
    for key, label in (("us", "Capital Rotation Report"), ("hk", "Hong Kong Rotation Report")):
        pdf_path = report_paths.get(f"pdf_{key}")
        if pdf_path is not None:
            caption = f"📊 *{label} — {asof.isoformat()} ({status_tag})*"
            res = telegram_send_document(pdf_path, caption=caption)
            telegram_results[key] = res
            summary[f"pdf_attachment_{key}"] = res

    # 7. Auto-commit reports (O1) — gated on (a) config flag, (b) at least one
    # PDF actually shipped to Telegram (so we don't commit reports the user
    # never saw), and (c) the standard skip-on-degraded / skip-on-failed checks
    # inside `commit_reports`. Off by default; user opts in via config.yaml.
    if cfg.reports.auto_commit:
        from .auto_commit import commit_reports
        any_telegram_ok = any(
            isinstance(r, dict) and "200" in str(r.get("telegram_doc", ""))
            for r in telegram_results.values()
        )
        # Heuristic for "degraded": the signals step was skipped with
        # insufficient_data. We treat any failed step in the report/alerts path
        # as a non-degraded failure (commit still blocked by `pipeline_failed`).
        degraded = any(
            (not s["ok"]) and "insufficient_data" in str(s.get("error", ""))
            for s in steps
        )
        if not any_telegram_ok:
            commit_res = {"status": "skipped:no_telegram_send", "sha": None, "files": []}
        else:
            headline_regime = ""
            for s in steps:
                if s["step"] == "regime" and isinstance(s.get("result"), dict):
                    r = s["result"]
                    headline_regime = f"Regime: {r.get('regime', '—')}"
                    if r.get("flipped"):
                        headline_regime += " (FLIP)"
                    break
            commit_res = commit_reports(
                asof,
                Path.cwd(),
                pipeline_failed=bool(failed),
                degraded=degraded,
                enabled=True,
                skip_on_degraded=cfg.reports.auto_commit_skip_on_degraded,
                body_summary=headline_regime,
            )
        summary["auto_commit"] = commit_res
        log.info("auto_commit: %s", commit_res.get("status"))

    return summary
