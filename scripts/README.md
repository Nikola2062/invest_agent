# Running TradingAgents (personal runner — `main.py`)

This is the **personal pre-market system** (overview digest + on-demand deep-dives +
Telegram bot), entry point `main.py` in the repo root. It is **not** the upstream
interactive CLI (`tradingagents` / `scripts/run_local.sh`), which is a separate tool.

All commands below are run from the repo root with the project's venv:
`/Users/sweden/Desktop/Pandora/Projects/TradingAgents`

---

## 0. Credentials — load `.env` first (one-time per shell)

`main.py` does **not** auto-load `.env`. It reads keys from the environment (or from
CLI flags). Load them once per shell:

```bash
set -a; source .env; set +a
```

Keys used: `DEEPSEEK_API_KEY` (LLM), `TELEGRAM_BOT_TOKEN` + `TELEGRAM_CHAT_ID`
(digest push + bot), `FRED_API_KEY` (optional — adds HY OAS + Sahm Rule to macro).

Alternative to sourcing: pass them inline —
`--deepseek-key sk-... --telegram-token <tok> --telegram-chat-id <id> --fred-key <key>`.

---

## 1. Daemon — the on-demand system (normal use)

```bash
set -a; source .env; set +a
.venv/bin/python main.py --schedule --run-now
```

- Pushes a **cheap, LLM-free overview** (~$0) to Telegram ~1h before each open
  (HK 08:30 HKT, US 08:30 ET — holiday/timezone-aware).
- **Auto-deep-dives** only held names that trip a deterministic alarm
  (drawdown-from-high / weak RS / risk-off), capped at `max_auto_deep`, deduped so a
  steady-state alarm doesn't re-run every fire.
- Serves the **on-demand bot** (see §3).
- `--run-now` also fires once immediately at startup (skip it to wait for the schedule).

Variants:

```bash
.venv/bin/python main.py --schedule --run-now --no-send   # build but don't push to Telegram
.venv/bin/python main.py --schedule --no-bot              # scheduled digests only, no bot
.venv/bin/python main.py --schedule --market us           # only the US group (default: all)
```

> Telegram allows **one** poller per bot token. Don't run two `--schedule` instances
> with the same token (the code guards this with a file lock and refuses the second).

---

## 2. One-shot — run specific names now (no daemon)

Uses the **full** pipeline on every ticker you pass (legacy `build_premarket` path;
the cheap overview + dedupe apply only in `--schedule` mode).

```bash
set -a; source .env; set +a
.venv/bin/python main.py --market us          # the US group from measure_config.json
.venv/bin/python main.py FIG NVDA             # ad-hoc tickers
.venv/bin/python main.py "^HSI"               # raw index (technical/news only — no fundamentals)
.venv/bin/python main.py --no-telegram FIG    # don't push the result
```

Output: combined HTML report + digest under `.tradingagents/reports/`.

---

## 3. Telegram bot — analyze a name on demand

The bot runs as part of `--schedule`. In your chat with the bot, send:

```
/us NVDA        → US market, English report
/hk 0700        → HK market, 繁體中文 report   (.HK is added for you → 0700.HK)
/us AMD zh      → US name, Chinese report      (append zh / en to switch language)
/hk 2800 en     → HK name, English report
us NVDA         → leading "/" is optional
```

Flow: bot acks immediately → runs the full pipeline for that one name (~5 min,
~$0.04–0.05) → sends back the HTML report as a document.

Guards: only `TELEGRAM_CHAT_ID` (or `telegram.allowlist_chat_ids`) is allowed;
120s cooldown between requests; 50/day cap (all configurable in §4).

---

## 4. Config — `scripts/measure_config.json`

Key knobs (CLI flags override these):

| Key | Meaning |
|---|---|
| `provider` / `deep_think_llm` / `quick_think_llm` | LLM provider + models (default DeepSeek) |
| `markets.<grp>.tickers` | ticker list per market group (`--market <grp>`) |
| `overview.mode` | `on_demand` (default) or `full` (legacy: deep-dive every ticker each fire) |
| `overview.drawdown_alarm_pct` | trip when a holding is this far below its recent **high** |
| `overview.drawdown_lookback_period` | window for the high (default `3mo`) |
| `overview.rs_alarm_score` | trip on relative strength this weak |
| `overview.max_auto_deep` | max auto deep-dives per fire |
| `overview.rerun_worsen_pct` / `rerun_after_days` | dedupe: re-run a name only if it worsens this much, or this long since last run |
| `schedule.runs` | per-market pre-open fire times (timezone + market calendar) |
| `telegram.cooldown_seconds` / `daily_cap` / `allowlist_chat_ids` | bot guards |

Book (holdings + watchlist) lives in `../positions.yaml`.

---

## Ticker notes (Hong Kong)

- HK codes are **4 digits** + `.HK`: `0700.HK`, `2800.HK`. A 5-digit form like
  `02800.HK` returns **no data** on yfinance — use `2800.HK`.
- Hang Seng exposure: `2800.HK` (Tracker Fund ETF, the tradeable proxy) via the bot,
  or `^HSI` (the raw index — one-shot CLI only, no company fundamentals).
- Symbols with `=` (e.g. `XAUUSD=X`, `GC=F`) are rejected by ticker validation; use an
  ETF proxy instead (e.g. `GLD` for gold).

---

## Output locations (`../.tradingagents/`)

- `reports/` — HTML reports + `*_digest.md`
- `cost_runs.json` — per-run token/cost log
- `outcomes.csv` — overlay verdicts + realized alpha (reflection loop)
- `triage_state.json` — per-name dedupe state (daemon)
- `logs/`, `cache/`, `memory/trading_memory.md` — runtime data
