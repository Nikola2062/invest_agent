"""Mode A: compose ONE unified digest from the three layers, in either language.

Block 1  — what changed in the market (US + HK): regime, top signals, index
           moves, and the macro gate gauges (VIX, Fear & Greed, CAPE, ...).
Block 2  — how your book should react: per held/watchlist name, the final action
           from the D1 overlay (stock verdict x macro+regime context).
Block 3  — action summary: the deduped to-do list.

`render_markdown` / `render_telegram` take lang="en" or "zh" (繁體中文); the same
complete information is rendered in each. Translation is deterministic (i18n.py).
"""
from __future__ import annotations

import html

import i18n
import settings
from adapters import macro_risk, rotation, stocks
import rules

_CFG = settings.CONFIG
_MV = _CFG["market_view"]


def _pct(x: float | None, mult: float = 100.0) -> str:
    if x is None:
        return "n/a"
    return f"{x * mult:+.1f}%"


def _num(x, nd: int = 2) -> str:
    try:
        return f"{float(x):,.{nd}f}"
    except (TypeError, ValueError):
        return str(x)


def gather(fresh_macro: bool = False) -> dict:
    """Collect raw inputs from the three layers (no formatting / no language)."""
    positions = stocks.read_positions()   # source of truth: stock_analysier/config
    book_held = positions["held"]
    book_watch = positions["watchlist"]
    hk_symbols = [b["symbol"] for b in book_held + book_watch if b["symbol"].endswith(".HK")]
    hk_symbols += [s for s in _MV.get("hk_indices", []) if s not in hk_symbols]

    macro = macro_risk.fetch(fresh=fresh_macro)
    rot = rotation.fetch(hk_symbols=hk_symbols)
    book_rows = stocks.read_book(book_held + book_watch)

    held_syms = {b["symbol"] for b in book_held}
    macro_level = macro.get("risk_level") if macro.get("available") else None
    regime = (rot.get("us_regime") or {}).get("label") if rot.get("available") else None

    verdicts = []
    for row in book_rows:
        latest = row["latest"]
        verdicts.append(
            rules.evaluate(
                symbol=row["symbol"],
                held=row["symbol"] in held_syms,
                base_tactical=(latest or {}).get("tactical_label"),
                base_reco=(latest or {}).get("if_not_held_recommendation"),
                macro_risk_level=macro_level,
                regime_label=regime,
            )
            | {"latest": latest, "market": row["market"]}
        )
    return {"macro": macro, "rotation": rot, "verdicts": verdicts,
            "macro_level": macro_level, "regime": regime}


# --- language-aware verdict text -------------------------------------------
def _verdict_action(v: dict, lang: str) -> str:
    if v["held"]:
        tac = i18n.tactical(v["final_tactical"], lang)
        if v["escalated"]:
            base = i18n.tactical(v["base_tactical"], lang)
            return (f"戰術：{tac}（由 {base} 升級）" if lang == "zh"
                    else f"tactical: {tac} (escalated from {base})")
        return f"戰術：{tac}" if lang == "zh" else f"tactical: {tac}"
    # not held
    if v["entry_blocked"]:
        was = i18n.reco(v["base_reco"], lang)
        return (f"暫緩進場（原為 {was}）" if lang == "zh"
                else f"entry blocked (was {v['base_reco']})")
    r = i18n.reco(v["final_reco"], lang) if v["final_reco"] else ("無" if lang == "zh" else "n/a")
    return f"建議：{r}" if lang == "zh" else f"recommendation: {r}"


def _verdict_reason(v: dict, macro_level, regime, lang: str) -> str:
    ml = i18n.risk_level(macro_level, lang) if macro_level else ("？" if lang == "zh" else "?")
    rg = i18n.regime(regime, lang) if regime else ("？" if lang == "zh" else "?")
    if v["danger"]:
        return (f"{i18n.t('macro_risk', lang)} {ml}／{i18n.t('regime', lang)} {rg}" if lang == "zh"
                else f"{i18n.t('macro_risk', lang)} {ml} / {i18n.t('regime', lang)} {rg}")
    return (f"{i18n.t('calm', lang)}（{i18n.t('macro', lang)} {ml}，{i18n.t('regime', lang)} {rg}）"
            if lang == "zh"
            else f"{i18n.t('calm', lang)} ({i18n.t('macro', lang)} {ml}, {i18n.t('regime', lang)} {rg})")


def render_markdown(data: dict, date_str: str, lang: str = "en") -> str:
    macro = data["macro"]
    rot = data["rotation"]
    L: list[str] = []
    L.append(f"# {i18n.t('title', lang)} — {date_str}")
    L.append("")

    # ---- Block 1: market change -------------------------------------------
    L.append(f"## {i18n.t('blk_market', lang)}")
    if macro.get("available"):
        lvl = i18n.risk_level(macro.get("risk_level"), lang)
        L.append(f"**{i18n.t('macro_gate', lang)}:** {lvl} — {i18n.macro_action(macro.get('action'), lang)}")
        g = macro.get("gauges", {})
        gauge_bits = []
        for key in ("vix", "fear_greed", "cape", "yield_curve", "hy_oas"):
            if key in g:
                disp = g[key].get("display") or g[key].get("value")
                gauge_bits.append(f"{i18n.gauge_label(key, lang)} {disp} [{i18n.light(g[key].get('light'), lang)}]")
        if gauge_bits:
            L.append(f"**{i18n.t('gauges', lang)}:** " + " · ".join(gauge_bits))
        sp = macro.get("sp500") or {}
        if sp:
            L.append(f"**{i18n.t('sp500', lang)}:** {_num(sp.get('current'))} "
                     f"({i18n.t('ath', lang)} {_num(sp.get('ath'))}, {_pct(sp.get('drawdown_pct'), 1)} {i18n.t('from_high', lang)})")
    else:
        L.append(f"_{macro.get('reason')}_")
    L.append("")

    if rot.get("available"):
        reg = rot.get("us_regime", {})
        L.append(f"**{i18n.t('us_regime', lang)}:** {i18n.regime(reg.get('label'), lang)} "
                 f"({i18n.t('confidence', lang)} {reg.get('confidence', 0):.2f}, "
                 f"{reg.get('days_in_regime', '?')}{i18n.t('days', lang)}) — {i18n.t('as_of', lang)} {rot.get('asof')}")
        sigs = rot.get("signals", {})
        topn = _MV.get("top_signals", 4)
        strongest = sorted(sigs.items(), key=lambda kv: abs(kv[1].get("score") or 0), reverse=True)[:topn]
        if strongest:
            bits = []
            for n, d in strongest:
                sc = "n/a" if d.get("score") is None else f"{d['score']:+.0f}"
                bits.append(f"{i18n.signal_name(n, lang)}: {sc}")
            L.append(f"**{i18n.t('strongest_signals', lang)}:** " + " · ".join(bits))
        movers = rot.get("us_movers", {})
        idx_bits = [f"{s} {_pct(movers[s]['r_d'])} 1d / {_pct(movers[s]['r_w'])} 5d / {_pct(movers[s]['r_m'])} 21d"
                    for s in _MV.get("us_indices", []) if s in movers]
        if idx_bits:
            L.append(f"**{i18n.t('us_indices', lang)}:** " + " | ".join(idx_bits))
        hk = rot.get("hk", {})
        if hk.get("regime_proxy"):
            rp = hk["regime_proxy"]
            br = _pct(rp.get("breadth")) if rp.get("breadth") is not None else "n/a"
            L.append(f"**{i18n.t('hk_regime', lang)}:** {i18n.regime(rp.get('label'), lang)} "
                     f"(HSI 21d {_pct(rp.get('benchmark_r_m'))}, {i18n.t('breadth', lang)} {br}) — {i18n.t('as_of', lang)} {hk.get('asof')}")
        hk_idx_bits = [f"{s} {_pct(m['r_d'])} 1d / {_pct(m['r_w'])} 5d / {_pct(m['r_m'])} 21d"
                       for s, m in (hk.get("movers") or {}).items()]
        if hk_idx_bits:
            L.append(f"**{i18n.t('hk_names', lang)}:** " + " | ".join(hk_idx_bits))
    else:
        L.append(f"_{rot.get('reason')}_")
    L.append("")

    # ---- Block 2: your book -----------------------------------------------
    L.append(f"## {i18n.t('blk_book', lang)}")
    ml = i18n.risk_level(data["macro_level"], lang) if data["macro_level"] else ("？" if lang == "zh" else "?")
    rg = i18n.regime(data["regime"], lang) if data["regime"] else ("？" if lang == "zh" else "?")
    L.append(f"_{i18n.t('context', lang)}: {i18n.t('macro', lang)} {ml}, {i18n.t('regime', lang)} {rg}._")
    L.append("")
    for v in data["verdicts"]:
        tag = i18n.t("held", lang) if v["held"] else i18n.t("watch", lang)
        latest = v.get("latest")
        stamp = latest["timestamp_utc"][:10] if latest and latest.get("timestamp_utc") else i18n.t("no_prior", lang)
        price = (latest or {}).get("current_price")
        action = _verdict_action(v, lang)
        reason = _verdict_reason(v, data["macro_level"], data["regime"], lang)
        line = (f"- **{v['symbol']}** ({v['market']}, {tag}) — {action}  "
                f"\n  _{reason}_; {i18n.t('last_analyzed', lang)} {stamp}")
        if price is not None:
            line += f", {i18n.t('px', lang)} {_num(price)}"
        L.append(line)
    L.append("")

    # ---- Block 3: action summary ------------------------------------------
    L.append(f"## {i18n.t('blk_actions', lang)}")
    todo = [v for v in data["verdicts"] if rules.needs_action(v)]
    if not todo:
        L.append(f"- {i18n.t('no_action', lang)}")
    else:
        for v in todo:
            L.append(f"- **{v['symbol']}**: {_verdict_action(v, lang)} — "
                     f"{_verdict_reason(v, data['macro_level'], data['regime'], lang)}")
    L.append("")
    L.append("---")
    L.append(f"_{i18n.t('footer', lang)}_")
    return "\n".join(L)


def render_telegram(data: dict, date_str: str, lang: str = "en") -> str:
    """Compact push version (<4096 chars), HTML-formatted for Telegram.

    HTML (not Markdown) so that dynamic content with underscores/asterisks
    (e.g. YELLOW_WATCH, WAIT_FOR_PRICE) can never break entity parsing. All
    interpolated values are HTML-escaped; only the <b> tags are literal.
    """
    e = html.escape
    macro = data["macro"]
    rot = data["rotation"]
    L = [f"📊 <b>{e(i18n.t('title', lang))}</b> {e(date_str)}"]
    if macro.get("available"):
        g = macro.get("gauges", {})
        vix = g.get("vix", {}).get("display") or g.get("vix", {}).get("value")
        fg = g.get("fear_greed", {}).get("display") or g.get("fear_greed", {}).get("value")
        L.append(f"{e(i18n.t('macro_gate', lang))}: <b>{e(i18n.risk_level(macro.get('risk_level'), lang))}</b> "
                 f"| VIX {e(str(vix))} | {e(i18n.gauge_label('fear_greed', lang))} {e(str(fg))}")
    if rot.get("available"):
        reg = rot.get("us_regime", {})
        L.append(f"{e(i18n.t('us_regime', lang))}: <b>{e(i18n.regime(reg.get('label'), lang))}</b> "
                 f"({reg.get('confidence', 0):.2f})")
        hk = (rot.get("hk") or {}).get("regime_proxy") or {}
        if hk:
            L.append(f"{e(i18n.t('hk_regime', lang))}: <b>{e(i18n.regime(hk.get('label'), lang))}</b>")
    L.append("")
    for v in data["verdicts"]:
        flag = "⚠️" if rules.needs_action(v) else "•"
        L.append(f"{flag} {e(v['symbol'])}: {e(_verdict_action(v, lang))}")
    todo = [v for v in data["verdicts"] if rules.needs_action(v)]
    if todo:
        L.append("")
        L.append(f"<b>{e(i18n.t('act', lang))}:</b> " + e(", ".join(v["symbol"] for v in todo)))
    return "\n".join(L)[:4096]
