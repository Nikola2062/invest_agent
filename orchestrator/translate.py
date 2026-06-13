"""Translate report markdown to Traditional Chinese (繁體中文) via DeepSeek.

Used for the FULL report and on-demand `/hk` replies (analyst prose + embedded
rotation reports are English). The compact digest is translated deterministically
in i18n.py and does NOT use this. Needs DEEPSEEK_API_KEY (passed via --deepseek-key);
callers fall back to English when `available()` is False.
"""
from __future__ import annotations

import os
import sys

import settings

_CFG = settings.CONFIG.get("translation", {}) or {}

_SYS_ZH = (
    "You are a professional financial translator. Translate the user's Markdown into "
    "Traditional Chinese (繁體中文, as used in Hong Kong / Taiwan).\n"
    "Rules:\n"
    "1. Preserve ALL Markdown structure exactly — headings (#), tables, lists, bold/italic, "
    "links, code blocks, blockquotes.\n"
    "2. Do NOT translate or alter: ticker symbols (e.g. FIG, 0700.HK, SPCX, SPY, QQQ), numbers, "
    "percentages, dates, currency codes (USD/HKD), and section numbers.\n"
    "3. Keep common financial acronyms in English: VIX, CAPE, DCF, ROIC, EV/EBITDA, P/E, OAS, "
    "ETF, HSI, YoY, RS, OBV, MoS.\n"
    "4. Translate prose, labels and headings into natural 繁體中文.\n"
    "5. Output ONLY the translated Markdown — no preamble, no explanation."
)


def backend() -> str:
    return "deepseek"


def available() -> bool:
    return bool(os.environ.get("DEEPSEEK_API_KEY"))


def _client():
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key:
        return None
    from openai import OpenAI
    base = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    return OpenAI(api_key=key, base_url=base)


def _chunks(md: str, max_chars: int = 6000) -> list[str]:
    lines = md.split("\n")
    chunks, cur, size = [], [], 0
    for ln in lines:
        if cur and size + len(ln) > max_chars and ln.startswith("#"):
            chunks.append("\n".join(cur))
            cur, size = [], 0
        cur.append(ln)
        size += len(ln) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks or [md]


def to_traditional_chinese(md: str) -> str | None:
    """Translate markdown to 繁體中文, or None if no DeepSeek key (caller falls
    back to English). Best-effort: a failed chunk keeps its English."""
    client = _client()
    if client is None:
        return None
    model = _CFG.get("deepseek_model") or os.environ.get("DEEPSEEK_MODEL_DEFAULT", "deepseek-chat")
    out = []
    chunks = _chunks(md)
    for i, ch in enumerate(chunks, 1):
        try:
            resp = client.chat.completions.create(
                model=model, temperature=0,
                messages=[{"role": "system", "content": _SYS_ZH}, {"role": "user", "content": ch}],
            )
            out.append(resp.choices[0].message.content)
        except Exception as e:
            print(f"[translate] chunk {i}/{len(chunks)} failed ({e}); keeping English", file=sys.stderr)
            out.append(ch)
    return "\n".join(out)


def translate_sections(mds: list[str]) -> list[str]:
    """Translate several sections (one API pass each). Same-length list; sections
    that can't be translated stay English."""
    return [(to_traditional_chinese(m) or m) for m in mds]
