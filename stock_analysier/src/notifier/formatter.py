"""Build compact, decision-only Telegram digests from AnalysisResult objects.

Constraints:
  - Telegram messages ≤ 4096 chars (we target ≤ 3500 for safety + readability)
  - Markdown formatting (avoid characters that break it: _, *, [, ], etc — escape when needed)
  - 4 push types: hk_morning_recap / hk_pre_close / us_pre_open / us_pre_close
"""
from __future__ import annotations

from datetime import datetime

from src.config.loader import load_portfolio
from src.models.schemas import AnalysisResult


# Telegram Markdown reserves these — but we use Markdown (v1), which is more lenient.
# We mostly avoid the problematic chars in our content rather than escape them.

def _level_emoji(label: str | None) -> str:
    return {
        "YELLOW_WATCH": "🟡",
        "ORANGE_TRIM": "🟠",
        "RED_DEFENSIVE": "🔴",
        "BLACK_EXIT": "⚫",
    }.get(label or "", "🟢")


def _signed(v: float, decimals: int = 1) -> str:
    sign = "+" if v >= 0 else ""
    return f"{sign}{v:.{decimals}f}"


def _money(v: float, currency: str) -> str:
    sym = {"USD": "$", "HKD": "HK$"}.get(currency, "")
    return f"{sym}{v:,.2f}"


def format_brief_digest(
    results: list[AnalysisResult],
    *,
    push_name: str,
    market_filter: str | None = None,
    purpose: str | None = None,
) -> str:
    """Brief digest for Telegram. One line per held position, then watchlist movers."""
    if market_filter:
        results = [r for r in results if r.market == market_filter]

    portfolio = load_portfolio()
    held_symbols = {h.symbol for h in portfolio.holdings}

    held = [r for r in results if r.symbol in held_symbols]
    watch = [r for r in results if r.symbol not in held_symbols]

    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    market_label = f"{market_filter} " if market_filter else ""
    header = f"*📊 {market_label}{purpose or push_name}*\n_{now}_"

    lines: list[str] = [header, ""]

    # --- HELD POSITIONS ---
    if held:
        lines.append("*Held positions*")
        for r in held:
            holding = portfolio.find(r.symbol)
            if not holding:
                continue
            pnl_pct = (r.current_price / holding.cost_basis_per_share - 1) * 100
            level = r.if_held.tactical.label
            emoji = _level_emoji(level)
            tactical_summary = (
                f"_{level}_" if level else "hold"
            )
            lines.append(
                f"{emoji} `{r.symbol}` {_money(r.current_price, r.currency)} "
                f"(cost {_money(holding.cost_basis_per_share, holding.currency)}, "
                f"P&L {_signed(pnl_pct)}%) → {tactical_summary}"
            )
            # Action line: immediate orders + first rebuy
            if r.if_held.immediate_orders:
                for o in r.if_held.immediate_orders[:1]:
                    lines.append(
                        f"   → {o.side} {o.quantity:.0f} @ "
                        f"{_money(o.limit_price or r.current_price, r.currency)} {o.order_type}"
                    )
            if r.if_held.rebuy_orders:
                bands = sorted(o.limit_price for o in r.if_held.rebuy_orders if o.limit_price)
                if bands:
                    lines.append(
                        f"   → rebuy band: {_money(bands[0], r.currency)} – {_money(bands[-1], r.currency)}"
                    )
            if r.if_held.tactical.hedge_recommended and r.hedge_plan and r.hedge_plan.candidates:
                rec = r.hedge_plan.candidates[r.hedge_plan.recommended_index]
                lines.append(f"   → hedge: short `{rec.instrument}` (corr {rec.correlation_90d:+.2f})" if rec.correlation_90d else f"   → hedge: short `{rec.instrument}`")
        lines.append("")

    # --- WATCHLIST MOVERS (only those with actionable signals) ---
    movers = []
    for r in watch:
        rec = r.if_not_held.recommendation
        if rec in ("BUY_NOW", "WAIT_FOR_PRICE"):
            movers.append(r)
    if movers:
        lines.append("*Watchlist actionable*")
        for r in movers[:6]:
            rec = r.if_not_held.recommendation
            mos = r.valuation.margin_of_safety_pct
            symbol_emoji = "🟢" if rec == "BUY_NOW" else "🔵"
            lines.append(
                f"{symbol_emoji} `{r.symbol}` {_money(r.current_price, r.currency)} "
                f"(MoS {_signed(mos)}%) → {rec}"
            )
            if r.if_not_held.entry_orders:
                o = r.if_not_held.entry_orders[0]
                lines.append(
                    f"   → BUY {o.quantity:.0f} @ {_money(o.limit_price, r.currency)} {o.order_type}"
                )
        lines.append("")

    # --- KEY MACRO SIGNALS ---
    all_signals = []
    for r in results:
        for s in r.risk.key_macro_signals[:3]:
            if s not in all_signals:
                all_signals.append(s)
    if all_signals:
        lines.append("*Key signals*")
        for s in all_signals[:5]:
            lines.append(f"• {s}")
        lines.append("")

    # --- KEY FORWARD CATALYSTS (next 30d) ---
    cats_seen: set[str] = set()
    cat_lines: list[str] = []
    for r in results:
        if r.forward_catalysts:
            for c in r.forward_catalysts.key_catalysts[:2]:
                key = f"{r.symbol}:{c.event}"
                if key in cats_seen:
                    continue
                cats_seen.add(key)
                date_part = f"{c.expected_date.isoformat()} " if c.expected_date else ""
                arrow = {"positive": "↑", "negative": "↓", "uncertain": "?"}.get(c.direction, "?")
                cat_lines.append(
                    f"• `{r.symbol}` {date_part}{arrow} {c.event} ({c.confidence})"
                )
    if cat_lines:
        lines.append("*Forward catalysts (30d)*")
        lines.extend(cat_lines[:6])
        lines.append("")

    if not held and not movers:
        lines.append("_No actionable items._")

    return "\n".join(lines).strip()


def format_thesis_break_alerts(alert_rows) -> str:
    """Format unpushed thesis-break alerts into an out-of-band Telegram message."""
    sev_emoji = {"critical": "🚨", "warning": "⚠️", "info": "ℹ️"}
    lines = ["*🚨 Thesis-break alerts*", ""]
    # Group by symbol for readability
    by_symbol: dict[str, list] = {}
    for row in alert_rows:
        by_symbol.setdefault(row["symbol"], []).append(row)
    for sym, rows in by_symbol.items():
        lines.append(f"*`{sym}`*")
        for r in rows:
            e = sev_emoji.get(r["severity"], "•")
            lines.append(f"{e} _{r['category']}_ — {r['summary']}")
        lines.append("")
    return "\n".join(lines).strip()
