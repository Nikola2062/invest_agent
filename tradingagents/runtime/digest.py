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


def _pnl_text(price: Optional[float], cost_basis: Optional[float]) -> str:
    if price is not None and cost_basis:
        return f"{(price - cost_basis) / cost_basis * 100.0:+.0f}% vs cost"
    return "n/a"


def _examples_footer(book: dict) -> list[str]:
    """Concrete on-demand bot examples, drawn from the user's own book so they're
    copy-pasteable. Falls back to sensible defaults for an empty book."""
    wl = book.get("watchlist") or {}
    us_list = wl.get("US") or []
    us = next((w["symbol"] for w in us_list if isinstance(w, dict) and w.get("symbol")), "NVDA")
    hk_candidates = [h.get("symbol", "") for h in (book.get("held") or [])]
    hk_candidates += [w.get("symbol", "") for w in (wl.get("HK") or []) if isinstance(w, dict)]
    hk = next((s[:-3] for s in hk_candidates if str(s).endswith(".HK")), "0700")
    return [
        "",
        "## 💬 Analyze any name on demand",
        f"  `/us {us}`  ·  `/hk {hk}`  ·  `/us {us} zh` (繁中)",
        "  _`/us <ticker>` = US (English) · `/hk <ticker>` = HK (.HK added for you, 繁中) "
        "· add `zh`/`en` to switch language._",
    ]


def build_overview_digest(date: str, macro: Optional[dict], book: dict,
                          prices: dict, ranking: Optional[list[dict]],
                          regime: Optional[str], alarms: list[dict],
                          running_syms: Optional[list[str]] = None) -> str:
    """Cheap, LLM-free pre-market overview.

    Shows only the deterministic quant layer (macro gate, regime, per-holding
    price action, cross-sectional RS rank) plus the names that tripped a
    deep-dive alarm. No per-name rating — those are run on demand via the bot.
    Pure string assembly; no network, no LLM.
    """
    lines = [f"# 🗓️ Pre-market overview — {date}", "", _macro_line(macro)]
    if regime:
        lines.append(f"📈 Regime: **{regime}**")
    if macro and macro.get("action"):
        lines.append(f"💰 {macro['action']}")
    lines.append("")

    held = book.get("held") or []
    if held:
        lines.append("## 📋 Your holdings")
        for h in held:
            sym = h.get("symbol")
            pnl = _pnl_text(prices.get(sym), h.get("cost_basis_per_share"))
            lines.append(f"  • {sym}: {pnl}")
        lines.append("")

    if ranking:
        lines.append("## 🏆 Opportunity rank (RS vs benchmark)")
        for r in ranking[:6]:
            lines.append(f"  {r['rank']}. {r['symbol']}  (RS {r['rs_score']:+.1f})")
        lines.append("")

    footer = _examples_footer(book)

    lines.append("## 🔎 Flagged for deep-dive")
    if not alarms:
        lines.append("  • Nothing tripped — reply `/us TICKER` or `/hk TICKER` to dig in.")
        return "\n".join(lines + footer)

    if running_syms is None:
        # No dedupe info supplied — legacy rendering (all flags auto-run).
        for a in alarms:
            lines.append(f"  ⚠ {a['symbol']}: {', '.join(a['reasons'])}")
        lines.append("")
        lines.append("_Full report auto-runs for flagged names._")
        return "\n".join(lines + footer)

    # Dedupe-aware: ▶ runs now, ⏸ already covered (steady-state alarm).
    run = set(running_syms)
    for a in alarms:
        mark = "▶" if a["symbol"] in run else "⏸"
        lines.append(f"  {mark} {a['symbol']}: {', '.join(a['reasons'])}")
    lines.append("")
    if run:
        lines.append("_▶ deep-diving now · ⏸ already covered (pull to refresh)._")
    else:
        lines.append("_All flagged names already covered._")
    return "\n".join(lines + footer)
