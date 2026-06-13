# orchestrator

A **non-destructive overlay** that combines the three sibling projects into one
investor workflow. It does **not** modify or import their code into its process —
it drives them through their own venvs and reads their on-disk data stores.

```
risk_analysis            →  macro gate     (risk level, VIX, Fear&Greed, CAPE …)
capital_rotation_analysier →  regime / flow  (US regime, 8 signals, US+HK moves)
stock_analysier          →  single-name    (per-ticker tactical verdict)
        └────────────── orchestrator ──────────────┘
                 ONE unified digest  +  on-demand bot
```

## Two modes

**Two report depths.** Each run produces (1) a **compact bilingual digest** — the readable
push message — and (2) a **full report** `reports/<date>_full.md`: the complete macro gate
table, capital_rotation's own full **US + HK** reports embedded verbatim (their 27-section
documents), and a full per-stock analysis for every held/watchlist name (fundamentals,
valuation, risk + scenarios, tactical/orders, technical, contrarian, devil's-advocate).
On `--send` the digest is the message and the full report is attached as a **PDF**
(a table of contents + a page break before each section: macro, rotation US, rotation HK,
and one per stock). The on-demand bot (`/us`,`/hk`) replies with the **full** single-stock
report as a PDF too.

**Translation** to 繁體中文 uses DeepSeek (needs `--deepseek-key`); without a key it
falls back to English.

**Bilingual:** the digest is rendered in every language listed in
`config.yaml → output.languages` (default `[en, zh]`) — the **same complete
information** in each. Files: `reports/<date>_digest.md` (English) and
`reports/<date>_digest.zh.md` (繁體中文). Both are pushed to Telegram when `--send`.
Translation is a deterministic lookup in `i18n.py` (no LLM).

**Mode A — daily digest (push).** One report in three blocks:
1. *What changed in the market* (US + HK): regime, strongest signals, index moves, macro gauges.
2. *How your book should react*: each held/watchlist name's final action = its own
   `stock_analysier` verdict **overlaid** with the macro level + regime (the D1 rules).
3. *Action summary*: the deduped to-do list.

**Mode B — interactive (pull).** Send an explicit market command into Telegram —
`/us <ticker>` or `/hk <ticker>` (e.g. `/us NVDA`, `/hk 0700`) → the bot runs
`stock_analysier`'s full pipeline for it → replies with a fresh deep-dive. The
market is always explicit (no guessing from the symbol); `.HK` is appended for
you on `/hk`. Guarded by a chat-id allowlist, per-chat cooldown, and daily cap.

## How the cross-layer feed works (D1)

The combine-logic lives in `rules.py` as a **post-hoc overlay** — the three projects'
own outputs are never mutated. Danger = macro level in `{HIGH, CRITICAL}` **or** regime
in the configured risk-off set. Under danger: held positions escalate one rung on the
tactical ladder, and new entries (`BUY_NOW`/`WAIT_FOR_PRICE`) are downgraded to `HOLD_OFF`.
All thresholds live in `config.yaml → rules`.

## Setup

ONE shared virtualenv at the repo root covers everything (run from the repo root):

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

Credentials are passed as **parameters** at launch — nothing needs to live on disk.
`main.py` exports them to the environment so the three projects (driven as
subprocesses with this same interpreter) inherit them: `stock_analysier` reads
DeepSeek/Finnhub from env, `capital_rotation_analysier` and `risk_analysis` read
FRED/Telegram from env. The sub-projects no longer keep their own venvs.

## Run — single entry point: `main.py` (at the repo root)

```bash
PY=.venv/bin/python

# Run ONCE now and push the unified digest to Telegram (default — no --schedule)
$PY main.py \
    --telegram-token <T> --telegram-chat-id <C> \
    --deepseek-key <D> --fred-key <F>

# Run on a SCHEDULE: digest pushed 1h before each market open (config.schedule.runs)
# + the on-demand bot, refreshing macro+rotation each run. Blocks; Ctrl-C to stop.
$PY main.py --schedule \
    --telegram-token <T> --telegram-chat-id <C> \
    --deepseek-key <D> --fred-key <F>

# Build a digest WITHOUT sending (no Telegram creds needed) — handy for testing
$PY main.py --no-send --no-fresh
```

Key flags: `--schedule` (run as a daemon at the config times; otherwise build once),
`--no-bot` (schedule: skip the bot), `--no-send` (build only), `--no-fresh` (skip
macro+rotation refresh), `--fresh` (one-time: force a refresh), `--run-now` (schedule:
also fire at startup).

Positions (held + watchlist) are read directly from **stock_analysier/config**
(`portfolio.yaml` holdings, `universe.yaml` watchlist) — edit those; the orchestrator
keeps no separate book and never overwrites them.

Schedule times live in `config.yaml → schedule.runs` — each `{time, timezone, calendar}`
fires in its own zone (DST handled), default **08:30 Asia/Hong_Kong [XHKG]** and
**08:30 America/New_York [XNYS]** (1h before each open). The `calendar`
(pandas-market-calendars) makes firing **holiday-aware**: a run is skipped when its
exchange is closed (weekends **and** holidays). `weekdays_only` is the fallback if a run
has no calendar.

### On-demand (Mode B) usage in Telegram
Once `--schedule` is running, message the bot — it replies with the **full** single-stock
report. The market is explicit and sets the language:
- `/us NVDA` → US, **English**
- `/hk 0700` → HK (`.HK` added), reply in **繁體中文 only** (HK is the Chinese indicator)
- override the language with a token: `/us NVDA zh` (Chinese) or `/hk 0700 en` (English)

Chinese replies are LLM-translated via DeepSeek (needs `--deepseek-key`); without a key the
bot falls back to English. Every incoming message and action is logged to the terminal.

### Freshness
`--schedule` refreshes the macro gate (`risk_analysis`) and capital rotation
(`capital_rotation_analysier` daily pipeline, with its own Telegram suppressed)
before each build. Per-stock verdicts come from `stock_analysier`'s `audit.sqlite`
(its last computed runs); refresh individual names live via the Mode B bot.

### Lower-level dev CLI (`run.py`)
`run.py` (in `orchestrator/`) exposes granular subcommands used during development —
`.venv/bin/python orchestrator/run.py daily [--send] [--fresh-macro]`, `… run.py serve`,
`… run.py once SYMBOL US`. `main.py` (repo root) is the supported launcher for normal use.

## Tests

```bash
cd orchestrator && ../.venv/bin/python -m pytest tests/ -q
```

## Files

| File | Role |
|---|---|
| `../main.py` | **single launcher** (repo root): `--schedule` (else one-time), credential params, scheduler + bot |
| `run.py` | lower-level dev CLI: `daily` / `serve` / `once` |
| `settings.py` | paths, config.yaml, .env loading, shared-venv interpreter |
| `config.yaml` | project paths, market view, D1 rules, schedule, Telegram limits (positions come from stock_analysier/config) |
| `adapters/macro_risk.py` | read newest `risk_analysis` report (+ `--fresh-macro` run) |
| `adapters/rotation.py` | read `rotation.duckdb` (US regime/signals + computed HK proxy) |
| `adapters/stocks.py` | read `audit.sqlite`; subprocess `analyze()` for fresh |
| `rules.py` | D1 overlay (pure, unit-tested) |
| `digest.py` | compose the compact bilingual digest + Telegram text (language-aware) |
| `full_report.py` | assemble the FULL report: macro + embedded rotation US/HK + per-stock |
| `stock_report.py` | render one stored AnalysisResult into a full markdown report |
| `i18n.py` | en/zh translation tables for the compact digest (deterministic, no LLM) |
| `translate.py` | full-report / `/hk` reply → 繁體中文 via DeepSeek (chunked, graceful fallback) |
| `pdf.py` | render markdown → web-grade PDF via **WeasyPrint** (cover, TOC w/ live page numbers, zebra tables, callouts, EN+CJK). Needs native libs — `brew install pango` |
| `telegram_bot.py` | send / send_document + two-way getUpdates listener (Mode B), with terminal logging |
| `_macro_runner.py` / `_stock_runner.py` | run a project's code via the shared venv (subprocess) |

## What is verified vs. what needs your creds
- ✅ Mode A end-to-end against live on-disk data (rotation + macro + stocks + D1 + digest).
- ✅ Single shared root `.venv`; all four projects import + run under it; `main.py` at repo root.
- ✅ D1 overlay + ticker parsing (11 unit tests).
- ⏳ Live Telegram send/receive — needs `--telegram-token`/`--telegram-chat-id`.
- ⏳ Live single-name `analyze()` — needs DeepSeek key + network (paid). Smoke test:
  `.venv/bin/python orchestrator/run.py once GOOGL US` (after creds set),
  or via the launcher: `.venv/bin/python main.py --mode once --deepseek-key <D> ...` then `/us GOOGL` in Telegram.
