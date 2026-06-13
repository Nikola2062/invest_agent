# investor_agent

A personal investing assistant that combines three independent tools into **one
workflow**: it tells you *what changed in the markets (US + HK)* and *what — if
anything — you should do about your holdings and watchlist*, then lets you pull a
deep-dive on any stock on demand via Telegram.

## What's inside

Three self-contained sub-projects, each still runnable on its own:

| Project | Layer | Answers |
|---|---|---|
| `risk_analysis` | **Macro gate** | Is the whole market dangerous? VIX, Fear & Greed, CAPE, yield curve… |
| `capital_rotation_analysier` | **Regime / flow** | Where is capital rotating? US regime + 8 signals; HK separately |
| `stock_analysier` | **Single-name** | Should I hold / trim / buy *this* stock? (23-agent pipeline) |

…tied together by a thin, **non-destructive** `orchestrator/` that drives them in
place and reads their data. Nothing about the three projects' own logic is modified.

```
risk_analysis ─┐
capital_rotation_analysier ─┤──>  orchestrator  ──>  unified daily digest (EN + 繁體中文)
stock_analysier ─┘                              └──>  two-way Telegram bot (on-demand)
```

## What you get

- **Mode A — daily digest:** one report in three blocks — *market change (US + HK)* →
  *how your book should react* → *action summary*. Rendered in **English and
  Traditional Chinese** (same content), written to `orchestrator/reports/` and
  optionally pushed to Telegram.
- **Mode B — on-demand:** message the Telegram bot `/us NVDA` or `/hk 0700` and it
  runs the full single-stock analysis and replies.

## Where your positions live

Positions are defined in **`stock_analysier/config/`** (the single source of truth):
- `portfolio.yaml` → `holdings:` — what you own (`symbol`, `market`, `shares`,
  `cost_basis_per_share`, `currency`, `purchase_date`, `notes`) and `cash:`
- `universe.yaml` → `watchlist:` (by market: `US:` / `HK:`) — names you track

Both the Streamlit dashboard and the orchestrator read these directly. The orchestrator
does **not** keep its own copy and never overwrites them.

## Getting started

### 1. System prerequisites (macOS / Linux)
You need **Python 3.12+** and the native libraries WeasyPrint uses to render the
report PDFs (pango/cairo), plus a **CJK font** so 繁體中文 PDFs render.

**macOS** (Homebrew):
```bash
brew install python@3.12 pango
# CJK fonts ship with macOS (PingFang / Arial Unicode) — nothing to install.
# Optional — only for the Mini App dashboard tunnel:
brew install cloudflared
```

**Linux** (Debian / Ubuntu):
```bash
sudo apt-get update
sudo apt-get install -y python3 python3-venv \
    libpango-1.0-0 libpangoft2-1.0-0 libharfbuzz0b \
    fonts-noto-cjk           # required — without it, 繁體中文 PDFs render as tofu (□)
# Optional — only for the Mini App dashboard tunnel:
#   grab cloudflared from https://github.com/cloudflare/cloudflared/releases
```
> Without pango, the text digest still builds but **PDF generation fails**. On
> Linux without `fonts-noto-cjk`, English works but Chinese PDFs show blank glyphs.

### 2. Create the environment (first time only)
**One** shared virtualenv at the repo root, installed once — it covers all three
sub-projects and the orchestrator:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
```

### 3. Set your positions
Edit `stock_analysier/config/portfolio.yaml` (holdings + cash) and
`stock_analysier/config/universe.yaml` (watchlist).

### 4. Try it offline (no credentials, no network calls)
Builds the digest from already-collected data and prints both languages:

```bash
.venv/bin/python main.py --no-send --no-fresh
```

Reports appear in `orchestrator/reports/<date>_digest.md` (EN) and `…_digest.zh.md` (繁體中文).

### 5. Go live
Credentials are passed as **parameters** (nothing is stored on disk). Get a Telegram
bot token + chat id from [@BotFather](https://t.me/BotFather); the DeepSeek and FRED
keys power live single-stock analysis and the macro gauges.

```bash
# run ONCE now and push the digest (default — no --schedule)
.venv/bin/python main.py \
    --telegram-token <T> --telegram-chat-id <C> --deepseek-key <D> --fred-key <F>

# run on a SCHEDULE: the digest is pushed 1h before each market opens
# (HK + US, timezone-aware) and the on-demand bot runs too. Blocks; Ctrl-C to stop.
.venv/bin/python main.py --schedule \
    --telegram-token <T> --telegram-chat-id <C> --deepseek-key <D> --fred-key <F>
```

The schedule times live in `orchestrator/config.yaml → schedule.runs` (default: 08:30
`Asia/Hong_Kong` and 08:30 `America/New_York` = 1h before each open; DST handled
automatically). Firing is **holiday-aware** — a run is skipped when its exchange (HKEX /
NYSE) is closed for a holiday or weekend. Edit them there.

Then, in Telegram: `/us NVDA`, `/hk 0700`, `/us SPCX`.

## Deployment (run it continuously)
`--schedule` runs as a long-lived process (pushes the digest 1h before each open and
serves the on-demand bot). Keep it alive across logout/reboot:

**macOS** — `nohup` (or wrap in a `launchd` plist):
```bash
nohup .venv/bin/python main.py --schedule \
  --telegram-token <T> --telegram-chat-id <C> --deepseek-key <D> --fred-key <F> \
  > orchestrator.log 2>&1 &
```

**Linux** — `systemd` (recommended). Create `/etc/systemd/system/investor-agent.service`:
```ini
[Unit]
Description=investor_agent orchestrator
After=network-online.target

[Service]
WorkingDirectory=/opt/investor_agent
EnvironmentFile=/opt/investor_agent/.deploy.env     # TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DEEPSEEK_API_KEY, FRED_API_KEY, FINNHUB_API_KEY
ExecStart=/opt/investor_agent/.venv/bin/python main.py --schedule
Restart=on-failure

[Install]
WantedBy=multi-user.target
```
```bash
sudo systemctl daemon-reload && sudo systemctl enable --now investor-agent
journalctl -u investor-agent -f          # watch logs
```
> **Credentials:** flags are visible in `ps`, so on a server prefer **environment
> variables** — `main.py` falls back to `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`,
> `DEEPSEEK_API_KEY`, `FRED_API_KEY`, `FINNHUB_API_KEY` from the environment when the
> matching `--flag` is omitted. Keep `.deploy.env` at `chmod 600`.

## More detail
- **`orchestrator/README.md`** — full runbook: every flag, the D1 cross-layer rules,
  the two-way Telegram bot, the Mini App dashboard, and the file map.

## Notes
- Everything runs locally; no data leaves your machine except API/Telegram calls you trigger.
- The three sub-projects can still be run standalone — the orchestrator only adds a layer on top.
