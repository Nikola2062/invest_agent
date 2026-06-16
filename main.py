"""TradingAgents — unified pre-market runner & launcher.

Two modes from ONE entry point:

  • one-shot (default): run one or more tickers (or a configured market group)
    through the multi-agent pipeline, build a combined HTML report + macro-risk
    context, apply the position-aware overlay, and optionally push to Telegram.

  • daemon (--schedule): push a compact digest 1h before each market open
    (holiday/timezone-aware) AND serve a two-way Telegram bot (/us NVDA, /hk 0700).

Credentials are passed as CLI parameters (or env fallback), never read from disk:
    --llm-key / --deepseek-key   LLM provider key (DeepSeek by default)
    --telegram-token / --telegram-chat-id
    --fred-key                   FRED key (HY OAS + Sahm Rule in the macro context)

Examples:
    python main.py --market us --llm-key sk-... --no-telegram
    python main.py --schedule --telegram-token 123:abc --telegram-chat-id 456 \
        --deepseek-key sk-... --fred-key ...

Security note: arguments are visible in `ps`; prefer env vars on shared hosts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
LOCAL_HOME = PROJECT_DIR / ".tradingagents"
SCRIPTS_DIR = PROJECT_DIR / "scripts"
DEFAULT_CONFIG_PATH = SCRIPTS_DIR / "measure_config.json"
POSITIONS_PATH = PROJECT_DIR / "positions.yaml"
sys.path.insert(0, str(SCRIPTS_DIR))

BUILTIN = {
    "tickers": ["NVDA"],
    "date": "today",
    "provider": "deepseek",
    "deep_think_llm": "deepseek-reasoner",
    "quick_think_llm": "deepseek-chat",
    "output_language": "English",
    "pricing": {"input_per_1m": 0.28, "output_per_1m": 0.42},
}

PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY", "google": "GOOGLE_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY", "xai": "XAI_API_KEY", "deepseek": "DEEPSEEK_API_KEY",
    "qwen": "DASHSCOPE_API_KEY", "qwen-cn": "DASHSCOPE_CN_API_KEY", "glm": "ZHIPU_API_KEY",
    "glm-cn": "ZHIPU_CN_API_KEY", "minimax": "MINIMAX_API_KEY", "minimax-cn": "MINIMAX_CN_API_KEY",
    "openrouter": "OPENROUTER_API_KEY", "ollama": None,
}


def parse_args(argv):
    p = argparse.ArgumentParser(description="TradingAgents pre-market runner & launcher.")
    p.add_argument("tickers", nargs="*", help="Ticker symbols (override config/market).")
    p.add_argument("--date", help='Analysis date YYYY-MM-DD or "today" (default: today).')
    p.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="Path to JSON config file.")
    p.add_argument("--market", nargs="+", metavar="GROUP",
                   help="Run market group(s) from config.markets (e.g. --market us, --market all).")
    p.add_argument("--provider", help="Override LLM provider from config (e.g. deepseek, openai).")
    # Credentials as parameters
    p.add_argument("--llm-key", help="API key for the LLM provider (sets the provider's key env var).")
    p.add_argument("--deepseek-key", help="DeepSeek API key (alias for --llm-key when provider=deepseek; "
                                          "also enables 繁體中文 translation).")
    p.add_argument("--telegram-token", help="Telegram bot token.")
    p.add_argument("--telegram-chat-id", help="Telegram chat id.")
    p.add_argument("--no-telegram", action="store_true", help="Skip Telegram delivery.")
    p.add_argument("--fred-key", help="FRED API key (enables HY OAS + Sahm Rule; falls back to FRED_API_KEY).")
    p.add_argument("--no-macro", action="store_true", help="Skip the market-wide macro risk context.")
    # Daemon mode
    p.add_argument("--schedule", action="store_true",
                   help="Run continuously: push the digest at each config.schedule.runs time "
                        "(1h before each open) + serve the on-demand bot. Without this, build once and exit.")
    p.add_argument("--no-bot", action="store_true", help="--schedule: do not run the on-demand Telegram bot.")
    p.add_argument("--no-send", action="store_true", help="Build the digest/report but do not push to Telegram.")
    p.add_argument("--run-now", action="store_true", help="--schedule: also fire the digest immediately at startup.")
    return p.parse_args(argv)


def load_config(path):
    cfg_path = Path(path)
    if not cfg_path.exists():
        print(f"[warn] config file not found at {cfg_path}; using built-in defaults.")
        return {}
    try:
        data = json.loads(cfg_path.read_text())
    except json.JSONDecodeError as e:
        raise SystemExit(f"[error] config file {cfg_path} is not valid JSON: {e}")
    if not isinstance(data, dict):
        raise SystemExit(f"[error] config file {cfg_path} must contain a JSON object.")
    return data


def resolve_date(cli_date, cfg_date):
    raw = cli_date if cli_date else cfg_date
    if not raw or str(raw).strip().lower() == "today":
        return datetime.now().strftime("%Y-%m-%d")
    raw = str(raw).strip()
    try:
        d = datetime.strptime(raw, "%Y-%m-%d")
    except ValueError:
        raise SystemExit(f"[error] invalid date {raw!r}; use YYYY-MM-DD or 'today'.")
    if d.date() > datetime.now().date():
        raise SystemExit(f"[error] analysis date {raw} cannot be in the future.")
    return raw


def _dedup_upper(raw):
    if isinstance(raw, str):
        raw = [raw]
    out, seen = [], set()
    for t in raw or []:
        t = str(t).strip().upper()
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def resolve_tickers(cli_tickers, group_tickers, cfg_tickers):
    tickers = _dedup_upper(cli_tickers or group_tickers or cfg_tickers or BUILTIN["tickers"])
    if not tickers:
        raise SystemExit("[error] no tickers specified (CLI, market, or config).")
    return tickers


def select_market(cfg, selected):
    """Return (label, group_tickers, market_slug) for the chosen market group(s)."""
    markets = cfg.get("markets") or {}
    selected = list(selected or [])
    if any(m.lower() == "all" for m in selected):
        selected = list(markets.keys())
    if not selected:
        return "Custom", None, None
    group_tickers, labels = [], []
    for m in selected:
        if m not in markets:
            raise SystemExit(f"[error] unknown market {m!r}; defined: {', '.join(markets) or '(none)'}.")
        for t in markets[m].get("tickers", []):
            if t not in group_tickers:
                group_tickers.append(t)
        labels.append(markets[m].get("label", m.upper()))
    return " + ".join(labels), group_tickers, "-".join(selected)


def apply_credentials(args, provider):
    """Export creds to env (params win over .env). Returns the resolved llm key."""
    # --deepseek-key is an alias for --llm-key when the provider is deepseek, and
    # always enables translation by exporting DEEPSEEK_API_KEY.
    llm_key = args.llm_key
    if args.deepseek_key:
        os.environ["DEEPSEEK_API_KEY"] = args.deepseek_key
        if provider.lower() == "deepseek" and not llm_key:
            llm_key = args.deepseek_key
    if args.telegram_token:
        os.environ["TELEGRAM_BOT_TOKEN"] = args.telegram_token
    if args.telegram_chat_id:
        os.environ["TELEGRAM_CHAT_ID"] = args.telegram_chat_id
    if args.fred_key:
        os.environ["FRED_API_KEY"] = args.fred_key
    key_env = PROVIDER_KEY_ENV.get(provider.lower())
    if key_env:
        if not llm_key and not os.environ.get(key_env):
            raise SystemExit(
                f"[error] provider {provider!r} requires --llm-key/--deepseek-key (sets {key_env})."
            )
        if llm_key:
            os.environ[key_env] = llm_key
    return llm_key


# --------------------------- pipeline + reporting ----------------------------

def run_pipeline(ticker, date, config, IN, OUT, cost_log):
    """Run one ticker through the graph; return a report_run dict (or None on error)."""
    from tradingagents.graph.trading_graph import TradingAgentsGraph
    from cli.stats_handler import StatsCallbackHandler

    def append_record(record):
        try:
            runs = json.loads(cost_log.read_text()) if cost_log.exists() else []
            if not isinstance(runs, list):
                runs = []
        except (json.JSONDecodeError, OSError):
            runs = []
        runs.append(record)
        cost_log.write_text(json.dumps(runs, indent=2))

    stats = StatsCallbackHandler()
    try:
        ta = TradingAgentsGraph(debug=False, config=config, callbacks=[stats])
        t0 = time.time()
        final_state, decision = ta.propagate(ticker, date)
        elapsed = time.time() - t0
    except Exception as e:
        print(f"  [error] {ticker} failed: {e}")
        append_record({"timestamp": datetime.now(timezone.utc).isoformat(),
                       "ticker": ticker, "date": date, "error": str(e)})
        return {"ticker": ticker, "error": str(e)}

    s = stats.get_stats()
    ti, to = s["tokens_in"], s["tokens_out"]
    cost = ti / 1e6 * IN + to / 1e6 * OUT
    print(f"  {ticker}: {decision}  ({ti + to:,} tok, ${cost:.4f}, {elapsed:.0f}s)")
    append_record({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": ticker, "date": date, "provider": config["llm_provider"],
        "elapsed_seconds": round(elapsed, 1),
        "llm_calls": s["llm_calls"], "tool_calls": s["tool_calls"],
        "tokens_in": ti, "tokens_out": to, "tokens_total": ti + to,
        "est_cost_usd": round(cost, 4), "decision": str(decision),
    })
    return {
        "ticker": ticker, "date": date, "decision": str(decision), "cost": cost,
        "tokens_total": ti + to,
        "stats": {"llm_calls": s["llm_calls"], "tool_calls": s["tool_calls"],
                  "tokens_in": ti, "tokens_out": to, "tokens_total": ti + to,
                  "est_cost_usd": round(cost, 4), "elapsed_seconds": round(elapsed, 1)},
        "state": final_state,
    }


def run_batch(tickers, date, config, pricing):
    """Run every ticker; return (report_runs, total_cost)."""
    IN, OUT = float(pricing["input_per_1m"]), float(pricing["output_per_1m"])
    cost_log = LOCAL_HOME / "cost_runs.json"
    cost_log.parent.mkdir(parents=True, exist_ok=True)
    report_runs, total_cost = [], 0.0
    for i, t in enumerate(tickers, 1):
        print(f"[{i}/{len(tickers)}] {t} on {date} ...")
        r = run_pipeline(t, date, config, IN, OUT, cost_log)
        if r and "error" not in r:
            report_runs.append(r)
            total_cost += r["cost"]
    return report_runs, total_cost


def build_macro(cfg, args):
    """Fetch the market-wide macro snapshot (or None)."""
    if args.no_macro:
        return None
    from macro_snapshot import build_snapshot
    macro_cfg = cfg.get("macro") or {}
    fred_key = args.fred_key or macro_cfg.get("fred_api_key") or os.environ.get("FRED_API_KEY")
    print("Fetching market risk context ...")
    try:
        macro = build_snapshot(fred_key=fred_key, manual_overrides=macro_cfg.get("manual_overrides"))
        t = macro["tally"]
        print(f"  Macro risk: {macro['risk_level']}  (🔴 {t['red']} · 🟡 {t['yellow']} · 🟢 {t['green']})")
        return macro
    except Exception as e:
        print(f"  [warn] macro snapshot failed, omitted: {e}")
        return None


def write_html_report(report_runs, date, market_slug, pricing, macro):
    from report_html import build_report
    reports_dir = LOCAL_HOME / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    mkt = f"{market_slug}_" if market_slug else ""
    slug = "-".join(r["ticker"] for r in report_runs).replace("/", "_").replace("=", "")
    out_path = reports_dir / f"report_{date}_{mkt}{slug}.html"
    out_path.write_text(build_report(report_runs, batch_date=date, pricing=pricing, macro=macro),
                        encoding="utf-8")
    return out_path


def build_premarket(tickers, date, cfg, pricing, args, *, market_slug=None, label="Custom"):
    """Full pre-market flow: batch -> macro -> overlay -> rank -> digest -> report.
    Returns (digest_text, html_path, report_runs, verdicts)."""
    from tradingagents.portfolio import positions as positions_mod
    from tradingagents.portfolio import relative_strength as rs
    from tradingagents.runtime import digest as digest_mod
    from tradingagents.runtime import reflect

    report_runs, total_cost = run_batch(tickers, date, _graph_config(cfg), pricing)
    macro = build_macro(cfg, args)
    macro_level = macro.get("risk_level") if macro else None

    book = positions_mod.load_positions(POSITIONS_PATH)
    held_syms = [h["symbol"] for h in book.get("held", [])]
    # Cross-sectional rank + coarse regime over held + this batch (best-effort, networked).
    rank_universe = list(dict.fromkeys(held_syms + [r["ticker"] for r in report_runs]))
    try:
        ranking = rs.fetch_and_rank(rank_universe)
        regime = rs.fetch_regime(rank_universe)
    except Exception as e:
        print(f"  [warn] ranking/regime unavailable: {e}")
        ranking, regime = [], None

    verdicts = positions_mod.overlay_for_runs(
        report_runs, book, macro_level=macro_level, regime=regime)

    html_path = write_html_report(report_runs, date, market_slug, pricing, macro) if report_runs else None
    digest_text = digest_mod.build_digest(date, macro, verdicts, ranking, regime)

    out_dir = LOCAL_HOME / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / f"{date}_digest.md").write_text(digest_text, encoding="utf-8")
    try:
        reflect.record_overlay_outcomes(verdicts, date, LOCAL_HOME / "outcomes.csv")
    except Exception as e:
        print(f"  [warn] outcome logging failed: {e}")
    print(f"\nBatch cost ${total_cost:.4f} · digest built ({len(verdicts)} names)")
    return digest_text, html_path, report_runs, verdicts


def send_digest(digest_text, html_path, label, date, cfg):
    """Push the digest text (EN + optional zh) and attach the HTML report."""
    from tradingagents.runtime import bot, translate
    token, chat_id = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat_id):
        print("[telegram] no token/chat-id — skipped.")
        return
    try:
        bot.send_message(token, chat_id, digest_text)
        langs = (cfg.get("output") or {}).get("languages", ["en", "zh"])
        if "zh" in langs and translate.available():
            zh = translate.to_traditional_chinese(digest_text)
            if zh:
                bot.send_message(token, chat_id, zh)
        if html_path is not None:
            bot.send_document(token, chat_id, html_path, caption=f"TradingAgents — {label} · {date}")
        print(f"[telegram] digest sent to chat {chat_id}.")
    except Exception as e:
        print(f"[telegram] delivery failed: {e}")


# ------------------------------- bot wiring ----------------------------------

def make_on_command(cfg, pricing, args):
    """Return an on_command(symbol, market, lang) for the Telegram bot: runs the
    pipeline for one name and returns {ok, path, caption} or {ok:False, error}."""
    def on_command(symbol, market, lang):
        date = datetime.now().strftime("%Y-%m-%d")
        report_runs, _ = run_batch([symbol], date, _graph_config(cfg), pricing)
        if not report_runs:
            return {"ok": False, "error": "pipeline produced no result"}
        macro = build_macro(cfg, args)
        path = write_html_report(report_runs, date, market.lower(), pricing, macro)
        return {"ok": True, "path": path, "caption": f"{symbol} ({market})"}
    return on_command


# ------------------------------- graph config --------------------------------

def _graph_config(cfg):
    from tradingagents.default_config import DEFAULT_CONFIG
    config = DEFAULT_CONFIG.copy()
    # Let the config file tune a whitelist of graph / cost knobs. Unknown keys
    # are ignored; anything absent falls back to the built-in default.
    for key in (
        "debate_report_mode",        # "compact" (cheap) | "full" (detailed)
        "debate_report_max_chars",   # per-report budget in compact mode
        "max_debate_rounds",
        "max_risk_discuss_rounds",
        "news_article_limit",
        "global_news_article_limit",
    ):
        if key in cfg:
            config[key] = cfg[key]
    return config


def _set_env_defaults(provider, deep_llm, quick_llm, output_language):
    os.environ.setdefault("TRADINGAGENTS_RESULTS_DIR", str(LOCAL_HOME / "logs"))
    os.environ.setdefault("TRADINGAGENTS_CACHE_DIR", str(LOCAL_HOME / "cache"))
    os.environ.setdefault("TRADINGAGENTS_MEMORY_LOG_PATH", str(LOCAL_HOME / "memory" / "trading_memory.md"))
    os.environ["TRADINGAGENTS_LLM_PROVIDER"] = provider
    os.environ["TRADINGAGENTS_DEEP_THINK_LLM"] = deep_llm
    os.environ["TRADINGAGENTS_QUICK_THINK_LLM"] = quick_llm
    os.environ["TRADINGAGENTS_OUTPUT_LANGUAGE"] = output_language


def main(argv):
    args = parse_args(argv)
    cfg = load_config(args.config)

    provider = args.provider or cfg.get("provider") or BUILTIN["provider"]
    deep_llm = cfg.get("deep_think_llm") or BUILTIN["deep_think_llm"]
    quick_llm = cfg.get("quick_think_llm") or BUILTIN["quick_think_llm"]
    output_language = cfg.get("output_language") or BUILTIN["output_language"]
    pricing = {**BUILTIN["pricing"], **(cfg.get("pricing") or {})}

    apply_credentials(args, provider)
    _set_env_defaults(provider, deep_llm, quick_llm, output_language)

    date = resolve_date(args.date, cfg.get("date"))
    send = not (args.no_send or args.no_telegram)
    tg_enabled = (cfg.get("telegram") or {}).get("enabled", False)

    # ----------------------------- daemon mode -----------------------------
    if args.schedule:
        from tradingagents.runtime import bot as botmod
        from tradingagents.runtime import scheduler

        # Scheduled digest covers all configured market groups by default.
        label, group_tickers, market_slug = select_market(cfg, args.market or ["all"])
        tickers = resolve_tickers(args.tickers, group_tickers, cfg.get("tickers"))
        if send and not (os.environ.get("TELEGRAM_BOT_TOKEN") and os.environ.get("TELEGRAM_CHAT_ID")):
            raise SystemExit("[error] --schedule send needs --telegram-token + --telegram-chat-id (or --no-send).")

        print(f"\n=== SCHEDULE mode ===  tickers: {', '.join(tickers)}  send={send}  bot={not args.no_bot}")

        def on_fire():
            d = datetime.now().strftime("%Y-%m-%d")
            digest_text, html_path, _, _ = build_premarket(
                tickers, d, cfg, pricing, args, market_slug=market_slug, label=label)
            if send:
                send_digest(digest_text, html_path, label, d, cfg)

        if not args.no_bot and os.environ.get("TELEGRAM_BOT_TOKEN"):
            tg = cfg.get("telegram") or {}
            allow = {str(c) for c in tg.get("allowlist_chat_ids", [])}
            if not allow and os.environ.get("TELEGRAM_CHAT_ID"):
                allow = {os.environ["TELEGRAM_CHAT_ID"]}

            def _serve():
                try:
                    botmod.serve(os.environ["TELEGRAM_BOT_TOKEN"], allow=allow,
                                 on_command=make_on_command(cfg, pricing, args),
                                 cooldown=tg.get("cooldown_seconds", 120),
                                 daily_cap=tg.get("daily_cap", 50),
                                 poll=tg.get("poll_timeout_seconds", 50))
                except botmod.PollerAlreadyRunning as e:
                    print(f"[bot] NOT started — {e}", file=sys.stderr)
            threading.Thread(target=_serve, name="telegram-bot", daemon=True).start()
            print("[bot] on-demand Telegram bot started")

        sched = cfg.get("schedule") or {}
        runs = sched.get("runs") or []
        if not runs:
            raise SystemExit("[error] --schedule but config.schedule.runs is empty.")
        try:
            scheduler.run_scheduler(runs, sched.get("weekdays_only", True), on_fire, run_now=args.run_now)
        except KeyboardInterrupt:
            print("\n[shutdown] stopped by user")
        return

    # ----------------------------- one-shot mode -----------------------------
    label, group_tickers, market_slug = select_market(cfg, args.market)
    tickers = resolve_tickers(args.tickers, group_tickers, cfg.get("tickers"))
    print(f"Provider={provider} deep={deep_llm} quick={quick_llm}")
    print(f"Market: {label}   Date: {date}   Tickers: {', '.join(tickers)}\n")

    digest_text, html_path, report_runs, _ = build_premarket(
        tickers, date, cfg, pricing, args, market_slug=market_slug, label=label)
    print("\n" + "=" * 60 + "\n" + digest_text + "\n" + "=" * 60)
    if html_path:
        print(f"HTML report: {html_path}")
    if send and tg_enabled and html_path is not None:
        send_digest(digest_text, html_path, label, date, cfg)


if __name__ == "__main__":
    main(sys.argv[1:])
