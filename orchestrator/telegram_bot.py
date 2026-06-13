"""Telegram transport: one-way send (Mode A push) + two-way listener (Mode B).

Mode B uses long-poll getUpdates (no public webhook/HTTPS needed for a local
daemon). On an incoming message it parses a ticker, runs stock_analysier's
analyze() in that project's venv (subprocess), and replies. Guarded by a
chat-id allowlist, a per-chat cooldown, and a daily cap.
"""
from __future__ import annotations

import re
import sys
import time

import requests

import settings
import stock_report
import translate
from adapters import stocks

_API = "https://api.telegram.org/bot{token}/{method}"
_CFG_TG = settings.CONFIG["telegram"]


def _log(msg: str) -> None:
    """Timestamped line to the terminal (flushed; the bot runs in a thread)."""
    print(f"[bot {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)

# Market is given EXPLICITLY by command: "/us NVDA" or "/hk 0700", with an
# optional trailing language token ("/us NVDA zh", "/hk 0700 en").
_CMD_RE = re.compile(r"^/?(us|hk)\s+([A-Za-z0-9.]{1,12})(?:\s+(\S+))?$", re.I)
_ZH_TOKENS = {"zh", "cn", "tc", "中", "中文", "繁", "繁中", "繁體", "繁體中文", "zh-hant"}
_EN_TOKENS = {"en", "eng", "english"}

HELP_TEXT = (
    "Send a market command:\n"
    "• /us <ticker> — US market (English), e.g. /us NVDA\n"
    "• /hk <ticker> — Hong Kong (繁體中文), e.g. /hk 0700\n"
    "The .HK suffix is added for you on /hk.\n"
    "Add a language to override: /us NVDA zh  or  /hk 0700 en"
)


def _call(method: str, token: str, **params):
    url = _API.format(token=token, method=method)
    r = requests.post(url, json=params, timeout=_CFG_TG.get("poll_timeout_seconds", 50) + 10)
    r.raise_for_status()
    return r.json()


def send(text: str, *, token: str | None = None, chat_id: str | None = None,
         parse_mode: str | None = None) -> dict:
    """Send a message. Default is PLAIN text (no parse_mode) so arbitrary content
    can never break entity parsing; pass parse_mode='HTML' for formatted content
    that has been escaped by the caller (e.g. the digest)."""
    token = token or settings.TELEGRAM_BOT_TOKEN
    chat_id = chat_id or settings.TELEGRAM_CHAT_ID
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set (pass --telegram-token/--telegram-chat-id)")
    # Telegram hard limit 4096; split conservatively on newlines.
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > 3900:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    if cur:
        chunks.append(cur)
    last = {}
    for ch in chunks:
        params = {"chat_id": chat_id, "text": ch, "disable_web_page_preview": True}
        if parse_mode:
            params["parse_mode"] = parse_mode
        last = _call("sendMessage", token, **params)
    return last


def send_document(path, *, token: str | None = None, chat_id: str | None = None,
                  caption: str | None = None) -> dict:
    """Upload a file (e.g. the full markdown report) as a Telegram document."""
    import pathlib
    token = token or settings.TELEGRAM_BOT_TOKEN
    chat_id = chat_id or settings.TELEGRAM_CHAT_ID
    if not token or not chat_id:
        raise RuntimeError("TELEGRAM_BOT_TOKEN / TELEGRAM_CHAT_ID not set")
    url = _API.format(token=token, method="sendDocument")
    p = pathlib.Path(path)
    with open(p, "rb") as fh:
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption[:1024]
        r = requests.post(url, data=data, files={"document": (p.name, fh)}, timeout=120)
    r.raise_for_status()
    return r.json()


def set_menu_button(url: str, *, text: str = "Dashboard", token: str | None = None) -> dict:
    """Set the bot's default menu button to a Web App (Mini App) opening `url`.

    Takes effect in PRIVATE chats with the bot (Mini App buttons don't work in
    groups). The url's domain must also be registered in BotFather (/setdomain).
    Re-run this whenever the tunnel URL changes.
    """
    token = token or settings.TELEGRAM_BOT_TOKEN
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    return _call("setChatMenuButton", token,
                 menu_button={"type": "web_app", "text": text, "web_app": {"url": url}})


def parse_command(text: str) -> tuple[str, str, str] | None:
    """Parse an explicit market command into (symbol, market, lang).

    Language defaults by market — HK => 'zh' (繁體中文), US => 'en' — and an
    optional trailing token overrides it.

    "/us NVDA"      -> ("NVDA", "US", "en")
    "/hk 0700"      -> ("0700.HK", "HK", "zh")
    "/us NVDA zh"   -> ("NVDA", "US", "zh")
    "/hk 0700 en"   -> ("0700.HK", "HK", "en")
    Anything else   -> None (caller shows HELP_TEXT).
    """
    text = (text or "").strip()
    m = _CMD_RE.match(text)
    if not m:
        return None
    market = m.group(1).upper()
    sym = m.group(2).upper()
    if sym.endswith(".HK"):
        sym = sym[:-3]
    if market == "HK":
        sym = f"{sym}.HK"
    lang = "zh" if market == "HK" else "en"  # default by market
    tok = (m.group(3) or "").strip()
    if tok:
        low = tok.lower()
        if low in _ZH_TOKENS or any(z in tok for z in ("中", "繁")):
            lang = "zh"
        elif low in _EN_TOKENS:
            lang = "en"
    return sym, market, lang


def serve() -> None:
    """Blocking long-poll loop. Ctrl-C to stop."""
    token = settings.TELEGRAM_BOT_TOKEN
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set (see orchestrator/.env)")
    allow = set(settings.allowlist_chat_ids())
    cooldown = _CFG_TG.get("cooldown_seconds", 120)
    daily_cap = _CFG_TG.get("daily_cap", 50)
    poll = _CFG_TG.get("poll_timeout_seconds", 50)

    last_used: dict[str, float] = {}
    used_today: dict[str, int] = {}
    day_key = [time.strftime("%Y-%m-%d")]
    offset = None
    me = _call("getMe", token)
    _log(f"bot online: @{me.get('result', {}).get('username')} | "
         f"allowlist={sorted(allow) or 'EMPTY (blocks all)'} | cooldown={cooldown}s cap={daily_cap}/day")

    while True:
        try:
            resp = _call("getUpdates", token, timeout=poll, offset=offset)
        except Exception as e:  # transient network: back off and retry
            _log(f"getUpdates error: {e}; retrying in 5s")
            time.sleep(5)
            continue

        for upd in resp.get("result", []):
            offset = upd["update_id"] + 1
            msg = upd.get("message") or upd.get("channel_post") or {}
            chat = msg.get("chat") or {}
            chat_id = str(chat.get("id", ""))
            text = msg.get("text", "")
            sender = (msg.get("from") or {}).get("username") or (msg.get("from") or {}).get("first_name") or "?"
            if not chat_id:
                continue
            _log(f"recv from chat={chat_id} ({sender}): {text!r}")
            if allow and chat_id not in allow:
                _log(f"  ignored — chat {chat_id} not in allowlist")
                continue

            # daily cap reset
            today = time.strftime("%Y-%m-%d")
            if today != day_key[0]:
                day_key[0] = today
                used_today.clear()

            parsed = parse_command(text)
            if parsed is None:
                _log("  unrecognized — sent help")
                send(HELP_TEXT, token=token, chat_id=chat_id)
                continue

            now = time.time()
            if now - last_used.get(chat_id, 0) < cooldown:
                wait = int(cooldown - (now - last_used.get(chat_id, 0)))
                _log(f"  cooldown — {wait}s remaining for chat {chat_id}")
                send(f"⏳ cooldown — try again in {wait}s.", token=token, chat_id=chat_id)
                continue
            if used_today.get(chat_id, 0) >= daily_cap:
                _log(f"  daily cap reached for chat {chat_id}")
                send("🚫 daily on-demand cap reached.", token=token, chat_id=chat_id)
                continue

            symbol, market, lang = parsed
            last_used[chat_id] = now
            used_today[chat_id] = used_today.get(chat_id, 0) + 1
            _log(f"  → analyzing {symbol} ({market}, lang={lang}) [{used_today[chat_id]}/{daily_cap} today]")
            ack = ("🔎 正在分析 " + symbol + "（" + market + "）…請稍候。" if lang == "zh"
                   else f"🔎 analyzing {symbol} ({market})… this takes a moment.")
            send(ack, token=token, chat_id=chat_id)

            res = stocks.analyze_fresh(symbol, market, persist=True)
            if not res.get("ok"):
                _log(f"  ✗ {symbol} failed: {res.get('error')}")
                send(f"❌ analysis failed for {symbol}: {res.get('error')}", token=token, chat_id=chat_id)
                continue

            full = res.get("full")
            if not full:
                _log(f"  ✓ {symbol} done — sending brief (no full payload)")
                send(res.get("telegram_text") or f"{symbol}: done.", token=token, chat_id=chat_id)
                continue

            report = stock_report.render_full(full)
            title = f"{symbol} ({market}) — full analysis"
            if lang == "zh":
                zh = translate.to_traditional_chinese(report)
                if zh:
                    report, title = zh, f"{symbol}（{market}）完整分析"   # reply ONLY in 繁體中文
                    _log(f"  ✓ {symbol} done — 繁體中文 ({len(report)} chars)")
                else:
                    _log(f"  ✓ {symbol} done — zh requested but no DeepSeek key; English")
            else:
                _log(f"  ✓ {symbol} done — en ({len(report)} chars)")

            # Deliver as a PDF document (nicer than a chunked text message).
            try:
                import pdf as pdfmod
                out_dir = settings.ORCH_DIR / settings.CONFIG["output"]["reports_dir"]
                out_dir.mkdir(parents=True, exist_ok=True)
                pdf_path = out_dir / f"ondemand_{symbol.replace('.', '_')}_{lang}.pdf"
                pdfmod.markdown_to_pdf(report, pdf_path, title=title, subtitle=symbol)
                send_document(pdf_path, token=token, chat_id=chat_id, caption=f"{symbol} ({market})")
                _log(f"  → sent PDF {pdf_path.name}")
            except Exception as e:
                _log(f"  PDF failed ({e}); falling back to text")
                send(report, token=token, chat_id=chat_id)   # send() chunks long text
