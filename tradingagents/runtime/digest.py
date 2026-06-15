"""Compose the compact pre-market digest from the four layers.

macro gate (macro_snapshot) + holder actions (position_overlay) + cross-sectional
rank (relative_strength) -> one readable markdown message. Pure string assembly;
no network, no LLM (Chinese is a separate translate pass on the output).
"""

from __future__ import annotations

from typing import Optional

from tradingagents.portfolio.position_overlay import needs_action

_HELD_EMOJI = {"HOLD": "🟢", "WATCH": "🟡", "TRIM": "🟠", "DEFENSIVE": "🔴", "EXIT": "⚫"}
_ENTRY_EMOJI = {"BUY_NOW": "🟢", "WAIT_FOR_PRICE": "🟡", "HOLD_OFF": "⚪", "AVOID": "🔴"}


def _macro_line(macro: Optional[dict]) -> str:
    if not macro:
        return "🌐 Market context: unavailable"
    t = macro.get("tally", {})
    sp = macro.get("sp500", {}) or {}
    sp_txt = ""
    if sp.get("drawdown_pct") is not None:
        sp_txt = f" · S&P {sp['drawdown_pct']:+.1f}% from ATH"
    return (f"🌐 Market: **{macro.get('risk_level', '?')}** risk "
            f"(🔴 {t.get('red', 0)} · 🟡 {t.get('yellow', 0)} · 🟢 {t.get('green', 0)}){sp_txt}")


def _verdict_line(v: dict) -> str:
    sym = v["symbol"]
    if v["held"]:
        emoji = _HELD_EMOJI.get(v["action"], "•")
        extra = []
        if v.get("unrealized_pnl_pct") is not None:
            extra.append(f"{v['unrealized_pnl_pct']:+.0f}%")
        if v.get("weight_pct") is not None:
            extra.append(f"{v['weight_pct']:.0f}% of book")
        tail = f" ({', '.join(extra)})" if extra else ""
        esc = " ⚠escalated" if v.get("escalated") else ""
        return f"  {emoji} {sym}: {v['rating']} → **{v['action']}**{tail}{esc}"
    emoji = _ENTRY_EMOJI.get(v["action"], "•")
    blk = " ⚠blocked" if v.get("entry_blocked") else ""
    return f"  {emoji} {sym}: {v['rating']} → **{v['action']}**{blk}"


def build_digest(date: str, macro: Optional[dict], verdicts: list[dict],
                 ranking: Optional[list[dict]] = None,
                 regime: Optional[str] = None) -> str:
    """Assemble the compact digest markdown."""
    lines = [f"# 🗓️ Pre-market digest — {date}", "", _macro_line(macro)]
    if regime:
        lines.append(f"📈 Regime: **{regime}**")
    if macro and macro.get("action"):
        lines.append(f"💰 {macro['action']}")
    lines.append("")

    held = [v for v in verdicts if v["held"]]
    entries = [v for v in verdicts if not v["held"]]

    if held:
        lines.append("## 📋 Your holdings")
        lines.extend(_verdict_line(v) for v in held)
        lines.append("")
    if entries:
        lines.append("## 👀 Watchlist / new entries")
        lines.extend(_verdict_line(v) for v in entries)
        lines.append("")

    if ranking:
        lines.append("## 🏆 Opportunity rank (RS vs benchmark)")
        for r in ranking[:6]:
            lines.append(f"  {r['rank']}. {r['symbol']}  (RS {r['rs_score']:+.1f})")
        lines.append("")

    # Action summary — the deduped to-do list.
    todo = [v for v in verdicts if needs_action(v)]
    lines.append("## ✅ Action summary")
    if todo:
        for v in todo:
            lines.append(f"  • {v['symbol']}: {v['action']}")
    else:
        lines.append("  • No action required — hold / monitor.")
    return "\n".join(lines)
