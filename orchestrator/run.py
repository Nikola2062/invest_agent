"""Orchestrator entry point.

  daily   Build the unified digest (Mode A). Reads on-disk data by default;
          --fresh-* flags re-run the underlying projects first.
            --send         push the compact digest to Telegram
            --fresh-macro  run risk_analysis once before reading
            --no-write     don't write the markdown report file
  serve   Run the two-way Telegram bot (Mode B). Blocks; Ctrl-C to stop.
  once    One-off on-demand analysis of a single ticker (prints + optional send).
            run.py once NVDA US [--send]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make this directory importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import settings  # noqa: E402
import digest  # noqa: E402
import telegram_bot  # noqa: E402
from adapters import stocks  # noqa: E402


def _today() -> str:
    return time.strftime("%Y-%m-%d")


def cmd_daily(args) -> int:
    data = digest.gather(fresh_macro=args.fresh_macro)
    date_str = _today()
    langs = settings.CONFIG["output"].get("languages", ["en"])
    out_dir = settings.ORCH_DIR / settings.CONFIG["output"]["reports_dir"]
    if not args.no_write:
        out_dir.mkdir(parents=True, exist_ok=True)

    for lang in langs:
        md = digest.render_markdown(data, date_str, lang=lang)
        print(f"\n===== [{lang}] =====\n{md}")
        if not args.no_write:
            suffix = "" if lang == "en" else f".{lang}"
            out_file = out_dir / f"{date_str}_digest{suffix}.md"
            out_file.write_text(md)
            print(f"\n[written] {out_file}", file=sys.stderr)
        if args.send:
            telegram_bot.send(digest.render_telegram(data, date_str, lang=lang), parse_mode="HTML")
            print(f"[sent] telegram digest ({lang})", file=sys.stderr)
    return 0


def cmd_serve(args) -> int:
    telegram_bot.serve()
    return 0


def cmd_once(args) -> int:
    res = stocks.analyze_fresh(args.symbol, args.market, persist=not args.no_persist)
    if not res.get("ok"):
        print(f"FAILED: {res.get('error')}", file=sys.stderr)
        if res.get("stderr"):
            print(res["stderr"], file=sys.stderr)
        return 1
    print(res.get("telegram_text", ""))
    if args.send:
        telegram_bot.send(res["telegram_text"])
        print("[sent]", file=sys.stderr)
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="orchestrator", description="Unified investor orchestrator")
    sub = p.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("daily", help="build the unified digest (Mode A)")
    d.add_argument("--send", action="store_true", help="push compact digest to Telegram")
    d.add_argument("--fresh-macro", action="store_true", help="run risk_analysis before reading")
    d.add_argument("--no-write", action="store_true", help="don't write the markdown file")
    d.set_defaults(func=cmd_daily)

    s = sub.add_parser("serve", help="run the two-way Telegram bot (Mode B)")
    s.set_defaults(func=cmd_serve)

    o = sub.add_parser("once", help="on-demand single-ticker analysis")
    o.add_argument("symbol")
    o.add_argument("market", choices=["US", "HK"])
    o.add_argument("--send", action="store_true")
    o.add_argument("--no-persist", action="store_true")
    o.set_defaults(func=cmd_once)

    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
