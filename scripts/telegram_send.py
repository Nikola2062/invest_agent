"""Minimal Telegram Bot API helpers (uses requests, already a project dep).

Credentials are read from environment variables (loaded from .env by the
caller): TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID.
"""
from __future__ import annotations

import os
from pathlib import Path

import requests

API = "https://api.telegram.org/bot{token}/{method}"
CAPTION_LIMIT = 1024
MESSAGE_LIMIT = 4096


def get_credentials():
    """Return (token, chat_id) from env, or (None, None) if unset."""
    return os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")


def send_document(token, chat_id, file_path, caption="", timeout=120):
    """Send a file as a document attachment with an optional caption."""
    file_path = Path(file_path)
    url = API.format(token=token, method="sendDocument")
    with file_path.open("rb") as fh:
        files = {"document": (file_path.name, fh, "text/html")}
        data = {"chat_id": chat_id, "caption": caption[:CAPTION_LIMIT]}
        resp = requests.post(url, data=data, files=files, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    if not body.get("ok"):
        raise RuntimeError(f"Telegram sendDocument failed: {body}")
    return body


def send_message(token, chat_id, text, timeout=60):
    """Send a plain-text message (truncated to Telegram's limit)."""
    url = API.format(token=token, method="sendMessage")
    data = {"chat_id": chat_id, "text": text[:MESSAGE_LIMIT]}
    resp = requests.post(url, data=data, timeout=timeout)
    resp.raise_for_status()
    body = resp.json()
    if not body.get("ok"):
        raise RuntimeError(f"Telegram sendMessage failed: {body}")
    return body


def build_caption(label, date, results, total_cost):
    """Compose a short caption: title + per-ticker rating + total cost.

    ``results`` is the list of {ticker, decision, cost, error?} dicts.
    """
    head = f"📡 TradingAgents — {label} · {date}\n"
    lines = []
    for r in results:
        if r.get("error"):
            lines.append(f"• {r['ticker']}: ⚠️ error")
        else:
            lines.append(f"• {r['ticker']}: {r.get('decision', '—')}")
    tail = f"\nTotal est. cost: ${total_cost:.4f}"
    caption = head + "\n".join(lines) + tail
    if len(caption) > CAPTION_LIMIT:  # keep the head + total, trim the middle
        caption = head + f"{len(results)} tickers" + tail
    return caption
