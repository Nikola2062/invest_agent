"""Telegram transport: one-way push (digest) + two-way on-demand listener.

Mode B uses long-poll getUpdates (no public webhook needed). On an incoming
``/us NVDA`` / ``/hk 0700`` it calls a supplied ``on_command(symbol, market, lang)``
callback (which runs the TradingAgents pipeline) and replies. Guarded by a chat-id
allowlist, a per-chat cooldown, a daily cap, and a single-poller flock.

Ported from investor_agent/orchestrator/telegram_bot.py; the project-specific
analysis is injected via the callback so this module stays pipeline-agnostic.
"""

from __future__ import annotations

import fcntl
import hashlib
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional

import requests

_API = "https://api.telegram.org/bot{token}/{method}"
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


class PollerAlreadyRunning(RuntimeError):
    pass


def _log(msg: str) -> None:
    print(f"[bot {time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _call(method: str, token: str, http_timeout: int = 60, **params):
    r = requests.post(_API.format(token=token, method=method), json=params, timeout=http_timeout)
    r.raise_for_status()
    return r.json()


def send_message(token: str, chat_id: str, text: str, parse_mode: Optional[str] = None) -> dict:
    """Send text, splitting on newlines under Telegram's 4096 limit."""
    chunks, cur = [], ""
    for line in text.split("\n"):
        if len(cur) + len(line) + 1 > 3900:
            chunks.append(cur)
            cur = ""
        cur += line + "\n"
    if cur:
        chunks.append(cur)
    last = {}
    for ch in chunks or [""]:
        params = {"chat_id": chat_id, "text": ch, "disable_web_page_preview": True}
        if parse_mode:
            params["parse_mode"] = parse_mode
        last = _call("sendMessage", token, **params)
    return last


def send_document(token: str, chat_id: str, path, caption: str = "") -> dict:
    p = Path(path)
    url = _API.format(token=token, method="sendDocument")
    with open(p, "rb") as fh:
        data = {"chat_id": chat_id}
        if caption:
            data["caption"] = caption[:1024]
        r = requests.post(url, data=data, files={"document": (p.name, fh)}, timeout=120)
    r.raise_for_status()
    return r.json()


def parse_command(text: str):
    """('NVDA','US','en') / ('0700.HK','HK','zh') / None."""
    m = _CMD_RE.match((text or "").strip())
    if not m:
        return None
    market = m.group(1).upper()
    sym = m.group(2).upper()
    if sym.endswith(".HK"):
        sym = sym[:-3]
    if market == "HK":
        sym = f"{sym}.HK"
    lang = "zh" if market == "HK" else "en"
    tok = (m.group(3) or "").strip()
    if tok:
        low = tok.lower()
        if low in _ZH_TOKENS or any(z in tok for z in ("中", "繁")):
            lang = "zh"
        elif low in _EN_TOKENS:
            lang = "en"
    return sym, market, lang


def _acquire_poll_lock(token: str):
    key = hashlib.sha256(token.encode()).hexdigest()[:16]
    lock_path = os.path.join(tempfile.gettempdir(), f"tradingagents_tgbot_{key}.lock")
    fh = open(lock_path, "w")
    try:
        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        fh.close()
        raise PollerAlreadyRunning(
            f"another getUpdates poller already runs for this bot token (lock {lock_path}). "
            f"Telegram allows only one — stop the other process first."
        )
    fh.write(f"{os.getpid()}\n")
    fh.flush()
    return fh


def serve(token: str, *, allow: set[str], on_command: Callable,
          cooldown: int = 120, daily_cap: int = 50, poll: int = 50) -> None:
    """Blocking long-poll loop. ``on_command(symbol, market, lang) -> dict`` with
    keys {ok, path?, caption?, text?, error?}. Ctrl-C to stop."""
    if not token:
        raise RuntimeError("TELEGRAM_BOT_TOKEN not set")
    _lock = _acquire_poll_lock(token)  # noqa: F841 — held for process lifetime

    last_used: dict[str, float] = {}
    used_today: dict[str, int] = {}
    day_key = [time.strftime("%Y-%m-%d")]
    offset = None
    me = _call("getMe", token)
    _log(f"online: @{me.get('result', {}).get('username')} | "
         f"allowlist={sorted(allow) or 'EMPTY (blocks all)'} | cooldown={cooldown}s cap={daily_cap}/day")

    while True:
        try:
            resp = _call("getUpdates", token, http_timeout=poll + 10, timeout=poll, offset=offset)
        except requests.exceptions.HTTPError as e:
            status = getattr(e.response, "status_code", None)
            _log(f"getUpdates HTTP {status}: {e}; backing off {'30' if status == 409 else '5'}s")
            time.sleep(30 if status == 409 else 5)
            continue
        except Exception as e:
            _log(f"getUpdates error: {e}; retrying in 5s")
            time.sleep(5)
            continue

        for upd in resp.get("result", []):
            offset = upd["update_id"] + 1
            msg = upd.get("message") or upd.get("channel_post") or {}
            chat_id = str((msg.get("chat") or {}).get("id", ""))
            text = msg.get("text", "")
            if not chat_id:
                continue
            _log(f"recv chat={chat_id}: {text!r}")
            if allow and chat_id not in allow:
                _log(f"  ignored — {chat_id} not in allowlist")
                continue

            today = time.strftime("%Y-%m-%d")
            if today != day_key[0]:
                day_key[0] = today
                used_today.clear()

            parsed = parse_command(text)
            if parsed is None:
                send_message(token, chat_id, HELP_TEXT)
                continue
            now = time.time()
            if now - last_used.get(chat_id, 0) < cooldown:
                wait = int(cooldown - (now - last_used.get(chat_id, 0)))
                send_message(token, chat_id, f"⏳ cooldown — try again in {wait}s.")
                continue
            if used_today.get(chat_id, 0) >= daily_cap:
                send_message(token, chat_id, "🚫 daily on-demand cap reached.")
                continue

            symbol, market, lang = parsed
            last_used[chat_id] = now
            used_today[chat_id] = used_today.get(chat_id, 0) + 1
            ack = (f"🔎 正在分析 {symbol}（{market}）…請稍候。" if lang == "zh"
                   else f"🔎 analyzing {symbol} ({market})… this takes a few minutes.")
            send_message(token, chat_id, ack)
            _log(f"  → analyzing {symbol} ({market}, {lang})")
            try:
                res = on_command(symbol, market, lang)
            except Exception as e:
                res = {"ok": False, "error": str(e)}
            if not res.get("ok"):
                send_message(token, chat_id, f"❌ analysis failed for {symbol}: {res.get('error')}")
                continue
            if res.get("path"):
                try:
                    send_document(token, chat_id, res["path"], caption=res.get("caption", symbol))
                    _log(f"  → sent {Path(res['path']).name}")
                    continue
                except Exception as e:
                    _log(f"  document send failed ({e}); falling back to text")
            send_message(token, chat_id, res.get("text") or f"{symbol}: done.")
