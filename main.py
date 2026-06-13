"""Single entry point for the unified investor orchestrator.

Start the whole system here. Credentials are passed as PARAMETERS (nothing
needs to live on disk) and are exported to the environment so the underlying
projects — driven as subprocesses — inherit them.

Examples
--------
Run ONCE now and push the digest to Telegram (default — no --schedule):
    python main.py \
        --telegram-token <T> --telegram-chat-id <C> \
        --deepseek-key <D> --fred-key <F>

Run on a SCHEDULE (digest 1h before each market open, from config + the bot):
    python main.py --schedule \
        --telegram-token <T> --telegram-chat-id <C> \
        --deepseek-key <D> --fred-key <F>

Build a digest without sending (no Telegram creds needed):
    python main.py --no-send
"""
from __future__ import annotations

import argparse
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

# Orchestrator modules live in ./orchestrator — make them importable.
sys.path.insert(0, str(Path(__file__).resolve().parent / "orchestrator"))


def _apply_creds(args) -> None:
    """Export passed credentials to the environment (params win over any .env).

    Done BEFORE importing settings/digest/telegram_bot so module-level reads and
    every child subprocess see them.
    """
    mapping = {
        "TELEGRAM_BOT_TOKEN": args.telegram_token,
        "TELEGRAM_CHAT_ID": args.telegram_chat_id,
        "DEEPSEEK_API_KEY": args.deepseek_key,
        "FRED_API_KEY": args.fred_key,
        "FINNHUB_API_KEY": args.finnhub_key,
    }
    for env_name, value in mapping.items():
        if value:
            os.environ[env_name] = value


def _build(send: bool, fresh: bool) -> dict:
    """Refresh (optional) → gather → render every configured language → write →
    optionally send. Returns {lang: markdown}."""
    import settings
    import digest
    import full_report
    import telegram_bot
    from adapters import rotation as rotation_adapter

    if fresh:
        ok, _ = rotation_adapter.run_daily()
        print(f"[fresh] rotation refresh ok={ok}", file=sys.stderr)

    data = digest.gather(fresh_macro=fresh)
    date_str = time.strftime("%Y-%m-%d")
    langs = settings.CONFIG["output"].get("languages", ["en"])

    out_dir = settings.ORCH_DIR / settings.CONFIG["output"]["reports_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) compact bilingual digest (the readable push message)
    rendered = {}
    for lang in langs:
        md = digest.render_markdown(data, date_str, lang=lang)
        rendered[lang] = md
        suffix = "" if lang == "en" else f".{lang}"
        (out_dir / f"{date_str}_digest{suffix}.md").write_text(md)
        if send:
            telegram_bot.send(digest.render_telegram(data, date_str, lang=lang), parse_mode="HTML")
            print(f"[sent] telegram digest ({lang})", file=sys.stderr)

    # 2) full combined report (macro + rotation US/HK + per-stock) -> markdown + PDF,
    #    with a table of contents and a page break before each section.
    import translate
    import pdf as pdfmod

    def _attach(path, label):
        if not send:
            return
        try:
            telegram_bot.send_document(path, caption=f"Full report {label} · {date_str}")
            print(f"[sent] telegram full report PDF ({label})", file=sys.stderr)
        except Exception as e:
            print(f"[full] PDF send failed ({label}): {e}", file=sys.stderr)

    def _first_heading(md, fallback):
        for ln in md.splitlines():
            if ln.lstrip().startswith("#"):
                return ln.lstrip("# ").strip() or fallback
        return fallback

    en_sections = full_report.build_sections(data, date_str)
    (out_dir / f"{date_str}_full.md").write_text(full_report.sections_to_markdown(en_sections, date_str))
    full_pdf = out_dir / f"{date_str}_full.pdf"
    pdfmod.render_report(en_sections, full_pdf, title="Investor — Full Report",
                         subtitle=f"{date_str} · US + HK · macro / rotation / per-stock",
                         toc_title="Contents")
    print(f"[full] wrote {full_pdf} ({len(en_sections)} sections)", file=sys.stderr)
    _attach(full_pdf, "EN")

    # Traditional Chinese full report (LLM-translated per section) when enabled + key present.
    if "zh" in langs and settings.CONFIG["output"].get("translate_full_report", True):
        if translate.available():
            print(f"[full] translating full report → 繁體中文 (backend={translate.backend()})…", file=sys.stderr)
            zh_mds = translate.translate_sections([s["md"] for s in en_sections])
            zh_sections = [{"title": _first_heading(zh, sec["title"]), "md": zh}
                           for sec, zh in zip(en_sections, zh_mds)]
            (out_dir / f"{date_str}_full.zh.md").write_text(
                full_report.sections_to_markdown(zh_sections, date_str))
            zh_pdf = out_dir / f"{date_str}_full.zh.pdf"
            pdfmod.render_report(zh_sections, zh_pdf, title="投資完整報告",
                                 subtitle=f"{date_str} · 美股 + 港股", toc_title="目錄")
            print(f"[full] wrote {zh_pdf}", file=sys.stderr)
            _attach(zh_pdf, "繁體中文")
        else:
            print("[full] zh full report skipped — no DeepSeek key (--deepseek-key)", file=sys.stderr)
    return rendered


def _trading_day(cal, d) -> bool:
    """True if date `d` is a trading day for the given market calendar."""
    return not cal.schedule(start_date=str(d), end_date=str(d)).empty


def _run_scheduler(runs: list, weekdays_only: bool, send: bool, fresh: bool, run_now: bool) -> None:
    """Fire the digest at each configured (time, timezone) — 1h before each open.

    Each run is evaluated in its OWN timezone so DST is handled correctly, fires
    at most once per local calendar day, and is HOLIDAY-aware: if the run names a
    `calendar` (pandas-market-calendars), it only fires on that exchange's trading
    days. Falls back to weekdays_only if no/invalid calendar.
    """
    from zoneinfo import ZoneInfo

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
        cal_name = r.get("calendar")
        if cal_name and mcal is not None:
            try:
                cal = mcal.get_calendar(cal_name)
            except Exception as e:
                print(f"[scheduler] bad calendar {cal_name!r} for '{r['label']}': {e} "
                      f"(falling back to weekdays_only)", file=sys.stderr)
        zones.append((r, tz, cal))

    sched_desc = ", ".join(
        f"{r['label']} {r['time']} {r['timezone']}"
        f"{' [' + r['calendar'] + ']' if cal is not None else ' [weekdays]'}"
        for r, _, cal in zones
    )
    print(f"[scheduler] runs: {sched_desc} | send={send} fresh={fresh}", file=sys.stderr)

    if run_now:
        try:
            _build(send, fresh)
            print("[scheduler] fired at startup", file=sys.stderr)
        except Exception as e:
            print(f"[scheduler] startup digest failed: {e}", file=sys.stderr)

    last_fired: dict = {}  # label -> local date already handled
    while True:
        for r, tz, cal in zones:
            now = datetime.now(tz)
            label = r["label"]
            if now.strftime("%H:%M") != r["time"] or last_fired.get(label) == now.date():
                continue
            last_fired[label] = now.date()  # mark handled (don't recheck this minute)

            if cal is not None:
                if not _trading_day(cal, now.date()):
                    print(f"[scheduler] '{label}' {now.date()} — {r['calendar']} closed (holiday/weekend), skipped",
                          file=sys.stderr)
                    continue
            elif weekdays_only and now.weekday() >= 5:  # 5=Sat, 6=Sun
                print(f"[scheduler] '{label}' {now.date()} — weekend, skipped", file=sys.stderr)
                continue

            try:
                _build(send, fresh)
                print(f"[scheduler] fired '{label}' {now.isoformat(timespec='seconds')}", file=sys.stderr)
            except Exception as e:
                print(f"[scheduler] '{label}' digest failed: {e}", file=sys.stderr)
        time.sleep(20)


def _print_startup(args) -> None:
    """Print a detailed summary of what's configured before doing any work."""
    import settings

    def mask(v):
        return "✓ set" if v else "—"

    from adapters import stocks
    sched = settings.CONFIG.get("schedule", {})
    positions = stocks.read_positions()   # source of truth: stock_analysier/config
    held, watch = positions["held"], positions["watchlist"]
    out_dir = settings.ORCH_DIR / settings.CONFIG["output"]["reports_dir"]
    refresh = (args.schedule and not args.no_fresh) or (args.fresh and not args.no_fresh)
    held_str = ", ".join(f"{h['symbol']} ({h['market']})" for h in held) or "—"
    watch_str = ", ".join(f"{w['symbol']} ({w.get('market', 'US')})" for w in watch) or "—"

    bar = "═" * 66
    P = lambda s: print(s, file=sys.stderr)
    P(bar)
    P("  investor_agent — unified investor orchestrator")
    P(bar)
    P(f"  mode             : {'SCHEDULE (daemon)' if args.schedule else 'ONE-TIME (build & exit)'}")
    P(f"  repo root        : {settings.ROOT}")
    P(f"  interpreter      : {settings.project_python()}")
    P(f"  languages        : {', '.join(settings.CONFIG['output'].get('languages', ['en']))}")
    P(f"  reports dir      : {out_dir}")
    P(f"  send to Telegram : {'yes' if not args.no_send else 'NO (--no-send)'}")
    P(f"  refresh data     : {'yes — macro + rotation' if refresh else 'no — reads latest on-disk'}")
    P(f"  credentials      : telegram {mask(args.telegram_token)} | chat {mask(args.telegram_chat_id)} | "
      f"deepseek {mask(args.deepseek_key)} | fred {mask(args.fred_key)} | finnhub {mask(args.finnhub_key)}")
    P(f"  positions (from stock_analysier/config): {len(held)} held, {len(watch)} watchlist")
    P(f"      held         : {held_str}")
    P(f"      watchlist    : {watch_str}")
    if args.schedule:
        P(f"  on-demand bot    : {'OFF (--no-bot)' if args.no_bot else 'ON'}")
        P(f"  immediate 1st run: {'YES (--run-now)' if args.run_now else 'no — waits for the next scheduled time'}")
        P(f"  weekdays_only    : {sched.get('weekdays_only', True)}")
        P("  schedule (digest fires 1h before each market open):")
        for r in sched.get("runs", []):
            P(f"      • {r.get('label')}: {r.get('time')} {r.get('timezone')} "
              f"[calendar {r.get('calendar', 'none → weekdays-only')}]")
    P(bar)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="investor-orchestrator", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    # --- mode ---
    p.add_argument("--schedule", action="store_true",
                   help="run continuously and push the digest at each config.schedule.runs time "
                        "(1h before each market open) + run the on-demand bot. Without this flag, "
                        "build once and exit.")
    # --- credentials (parameters, not on disk) ---
    p.add_argument("--telegram-token", help="Telegram bot token (push + on-demand bot)")
    p.add_argument("--telegram-chat-id", help="Telegram chat id to push to / allow")
    p.add_argument("--deepseek-key", help="DeepSeek API key (live single-name analysis + rotation narrative)")
    p.add_argument("--fred-key", help="FRED API key (macro gauges)")
    p.add_argument("--finnhub-key", help="Finnhub API key (stock fundamentals/catalysts)")
    # --- options ---
    p.add_argument("--no-bot", action="store_true", help="--schedule: do NOT run the on-demand Telegram bot")
    p.add_argument("--no-send", action="store_true", help="build the digest but do not push to Telegram")
    p.add_argument("--no-fresh", action="store_true", help="do NOT refresh macro+rotation before building")
    p.add_argument("--fresh", action="store_true", help="one-time: refresh macro+rotation before building")
    p.add_argument("--run-now", action="store_true", help="--schedule: also fire the digest immediately at startup")
    args = p.parse_args(argv)

    _apply_creds(args)

    # Imports happen AFTER creds are exported.
    import settings  # noqa: E402
    import telegram_bot  # noqa: E402

    _print_startup(args)

    # Positions are read directly from stock_analysier/config (single source of
    # truth) — the orchestrator no longer maintains or syncs its own book.

    send = not args.no_send
    if send and not (os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID")):
        p.error("--telegram-token and --telegram-chat-id are required unless --no-send is given")

    # --- one-time run (default) ---
    if not args.schedule:
        fresh = args.fresh and not args.no_fresh
        rendered = _build(send, fresh)
        for lang, md in rendered.items():
            print(f"\n===== [{lang}] =====\n{md}")
        return 0

    # --- scheduled daemon ---
    fresh = not args.no_fresh  # scheduled runs refresh by default
    sched = settings.CONFIG.get("schedule", {})
    runs = sched.get("runs", [])
    if not runs:
        p.error("--schedule given but config.schedule.runs is empty")

    if not args.no_bot:
        if not os.environ.get("TELEGRAM_BOT_TOKEN"):
            p.error("--telegram-token is required to run the on-demand bot (or pass --no-bot)")
        t = threading.Thread(target=telegram_bot.serve, name="telegram-bot", daemon=True)
        t.start()
        print("[bot] on-demand Telegram bot started", file=sys.stderr)

    try:
        _run_scheduler(runs, sched.get("weekdays_only", True), send, fresh, args.run_now)
    except KeyboardInterrupt:
        print("\n[shutdown] stopped by user", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
