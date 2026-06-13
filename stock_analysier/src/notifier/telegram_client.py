"""Minimal Telegram Bot client — direct HTTP, no extra dependencies."""
from __future__ import annotations

import logging
from typing import Literal

import requests

from src.config.loader import load_secrets

log = logging.getLogger(__name__)

API_BASE = "https://api.telegram.org"
MAX_MESSAGE_LEN = 4000  # below Telegram's 4096 to leave room for parse_mode safety


class TelegramError(Exception):
    pass


def _telegram_url(method: str) -> str:
    secrets = load_secrets()
    if not secrets.telegram.bot_token:
        raise TelegramError("Telegram bot_token not set. Edit config/secrets.yaml.")
    return f"{API_BASE}/bot{secrets.telegram.bot_token}/{method}"


def _chat_id() -> str:
    secrets = load_secrets()
    if not secrets.telegram.chat_id:
        raise TelegramError("Telegram chat_id not set. Edit config/secrets.yaml.")
    return secrets.telegram.chat_id


def _split_message(text: str, limit: int = MAX_MESSAGE_LEN) -> list[str]:
    if len(text) <= limit:
        return [text]
    chunks = []
    while text:
        # Break on a newline boundary if possible
        if len(text) <= limit:
            chunks.append(text)
            break
        cut = text.rfind("\n", 0, limit)
        if cut < limit // 2:
            cut = limit
        chunks.append(text[:cut])
        text = text[cut:].lstrip("\n")
    return chunks


def send_message(
    text: str,
    parse_mode: Literal["Markdown", "MarkdownV2", "HTML", None] = "Markdown",
    disable_preview: bool = True,
    chat_id_override: str | None = None,
) -> None:
    """Send a Telegram message (auto-split if too long)."""
    chat_id = chat_id_override or _chat_id()
    url = _telegram_url("sendMessage")
    chunks = _split_message(text)
    for i, chunk in enumerate(chunks):
        payload = {
            "chat_id": chat_id,
            "text": chunk,
            "disable_web_page_preview": disable_preview,
        }
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            r = requests.post(url, json=payload, timeout=15)
            r.raise_for_status()
        except requests.RequestException as e:
            # If markdown parsing fails, retry without parse_mode
            if parse_mode and r is not None and r.status_code == 400:
                payload.pop("parse_mode", None)
                r2 = requests.post(url, json=payload, timeout=15)
                r2.raise_for_status()
                continue
            raise TelegramError(f"sendMessage failed (chunk {i+1}/{len(chunks)}): {e}") from e


def test_connection() -> dict:
    """Calls getMe to validate the bot token."""
    url = _telegram_url("getMe")
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    return r.json()
