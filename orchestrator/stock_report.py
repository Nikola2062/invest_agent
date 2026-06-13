"""Render a stock_analysier AnalysisResult (as a dict from full_result_json or
model_dump) into a FULL markdown report — fundamentals, valuation, risk,
scenarios, tactical/orders, technical, contrarian, and the devil's-advocate
review. Defensive: every field is optional and missing sections are skipped.

The prose fields (rationales, summaries) are the analyst model's own output and
are in English; this renderer does not translate them.
"""
from __future__ import annotations


def _num(x, nd: int = 2) -> str:
    try:
        return f"{float(x):,.{nd}f}"
    except (TypeError, ValueError):
        return "n/a" if x is None else str(x)


def _pct(x, nd: int = 1) -> str:
    try:
        return f"{float(x):+.{nd}f}%"
    except (TypeError, ValueError):
        return "n/a"


def _line(label: str, value, fmt=None) -> list[str]:
    if value is None or value == "":
        return []
    if fmt:
        value = fmt(value)
    return [f"- **{label}:** {value}"]


def _kv_block(d: dict) -> list[str]:
    """Generic fallback for an unknown dict (e.g. an order)."""
    out = []
    for k, v in d.items():
        if v in (None, "", [], {}):
            continue
        out.append(f"  - {k}: {v}")
    return out


def render_full(r: dict) -> str:
    L: list[str] = []
    sym = r.get("symbol", "?")
    L.append(f"# {sym} ({r.get('market', '?')}) — full analysis")
    L.append(f"_As of {str(r.get('timestamp_utc', ''))[:19]}Z · price {r.get('currency', '')} {_num(r.get('current_price'))}_")
    L.append("")

    pos = r.get("position")
    if pos:
        shares = pos.get("shares")
        cb = pos.get("cost_basis_per_share")
        L.append(f"**Position:** {_num(shares, 0)} sh @ {_num(cb)} {pos.get('currency', '')}"
                 + (f" · {pos.get('notes')}" if pos.get("notes") else ""))
        try:
            pnl = (float(r["current_price"]) / float(cb) - 1) * 100
            L.append(f"**Unrealized:** {pnl:+.1f}% vs cost")
        except (TypeError, ValueError, ZeroDivisionError, KeyError):
            pass
        L.append("")

    comp = r.get("competence")
    if comp:
        L.append(f"**Competence gate:** {comp.get('verdict')} — {comp.get('reasoning', '')}")
        L.append("")

    # --- Fundamental ---
    f = r.get("fundamental")
    if f:
        L.append("## Fundamental")
        if f.get("thesis_one_liner"):
            L.append(f"> {f['thesis_one_liner']}")
        L += _line("Quality score", f.get("quality_score"), lambda v: f"{_num(v,1)} / 10")
        L += _line("Moat", f"{f.get('moat_strength', '')} — {f.get('moat_assessment', '')}".strip(" —"))
        L += _line("Balance sheet", f.get("balance_sheet_health"))
        L += _line("Growth outlook", f.get("growth_outlook"))
        L += _line("Capital allocation", f.get("capital_allocation"))
        L += _line("ROIC", f.get("roic_pct"), lambda v: _pct(v))
        L += _line("Gross margin", f.get("gross_margin_pct"), lambda v: _pct(v))
        L += _line("Operating margin", f.get("operating_margin_pct"), lambda v: _pct(v))
        L += _line("Debt / equity", f.get("debt_to_equity"), lambda v: _num(v))
        flags = f.get("red_flags") or []
        if flags:
            L.append("- **Red flags:**")
            L += [f"  - {x}" for x in flags]
        L.append("")

    # --- Valuation ---
    v = r.get("valuation")
    if v:
        L.append("## Valuation")
        L.append(f"- **Intrinsic range:** {_num(v.get('intrinsic_low'))} / "
                 f"**{_num(v.get('intrinsic_base'))}** / {_num(v.get('intrinsic_high'))} "
                 f"({v.get('confidence', '?')} confidence)")
        L += _line("Margin of safety", v.get("margin_of_safety_pct"), lambda x: _pct(x))
        L += _line("DCF value", v.get("dcf_value"), lambda x: _num(x))
        L += _line("Multiples value", v.get("multiples_value"), lambda x: _num(x))
        L += _line("P/E", v.get("pe_ratio"), lambda x: _num(x))
        L += _line("EV/EBITDA", v.get("ev_to_ebitda"), lambda x: _num(x))
        L += _line("Notes", v.get("methodology_notes"))
        L.append("")

    # --- Risk ---
    rk = r.get("risk")
    if rk:
        L.append("## Risk")
        L += _line("Realized vol (annualized)", rk.get("realized_vol_annualized_pct"), lambda x: _pct(x, 0))
        dd = rk.get("drawdown_probabilities") or {}
        if dd:
            L.append("- **Drawdown probabilities (" + str(rk.get("horizon_days", "?")) + "d):** "
                     + " · ".join(f"≥{k}%: {float(val) * 100:.0f}%" for k, val in dd.items()))
        for s in rk.get("scenarios") or []:
            L.append(f"  - *{s.get('name')}* (p={s.get('probability')}): "
                     f"ret {_pct(s.get('expected_return_pct'))}, dd {_pct(s.get('expected_drawdown_pct'))} — {s.get('rationale', '')}")
        L.append("")

    # --- Forward scenarios ---
    fs = r.get("forward_scenarios")
    if fs:
        L.append("## Forward scenarios")
        L += _line("Prob-weighted target", fs.get("probability_weighted_target"), lambda x: _num(x))
        L += _line("Expected return", fs.get("expected_return_pct"), lambda x: _pct(x))
        if fs.get("summary"):
            L.append(f"> {fs['summary']}")
        for s in fs.get("scenarios") or []:
            L.append(f"  - *{s.get('name')}* (p={s.get('probability')}): "
                     f"base {_num(s.get('target_price_base'))} ({_pct(s.get('return_pct_base'))}) — {s.get('rationale', '')}")
        L.append("")

    # --- Forward catalysts ---
    fc = r.get("forward_catalysts")
    if fc:
        L.append("## Catalysts (next " + str(fc.get("horizon_days", "?")) + "d)")
        L += _line("Sentiment", fc.get("sentiment_summary"))
        L += _line("Sentiment score", fc.get("sentiment_score"))
        for c in fc.get("key_catalysts") or []:
            L.append(f"  - {c.get('event')} (~{c.get('expected_date')}, {c.get('direction')}, "
                     f"{_pct(c.get('expected_magnitude_pct'))}) — {c.get('rationale', '')}")
        L.append("")

    # --- Decision: held vs not held ---
    ih = r.get("if_held") or {}
    tac = ih.get("tactical") or {}
    if tac:
        L.append("## Tactical (held)")
        L.append(f"- **Level {tac.get('level')} — {tac.get('label')}** → {tac.get('action')}")
        L += _line("Trim % of position", tac.get("trim_pct_of_position"))
        if tac.get("rebuy_band_low") is not None:
            L.append(f"- **Rebuy band:** {_num(tac.get('rebuy_band_low'))} – {_num(tac.get('rebuy_band_high'))}")
        L += _line("Hedge recommended", tac.get("hedge_recommended"))
        L += _line("Rationale", tac.get("rationale"))
        for orders, title in [(ih.get("immediate_orders"), "Immediate orders"),
                              (ih.get("rebuy_orders"), "Rebuy orders")]:
            if orders:
                L.append(f"- **{title}:**")
                for o in orders:
                    L += _kv_block(o) if isinstance(o, dict) else [f"  - {o}"]
        L.append("")

    inh = r.get("if_not_held")
    if inh:
        L.append("## If not held")
        L.append(f"- **Recommendation:** {inh.get('recommendation')}")
        L += _line("Rationale", inh.get("rationale"))
        for o in inh.get("entry_orders") or []:
            L += _kv_block(o) if isinstance(o, dict) else [f"  - {o}"]
        L.append("")

    # --- Technical ---
    t = r.get("technical")
    if t:
        L.append("## Technical")
        L.append(f"- **Composite signal:** {t.get('composite_signal')} — {t.get('composite_rationale', '')}")
        st = t.get("structure") or {}
        if st:
            L.append(f"- **Structure:** {st.get('trend')} / {st.get('stage')} "
                     f"(conf {st.get('confidence')}); swing {_num(st.get('last_swing_low'))}–{_num(st.get('last_swing_high'))}")
        for key in ("volume", "cost_basis", "relative_strength", "price_map"):
            sub = t.get(key)
            if isinstance(sub, dict):
                bits = ", ".join(f"{k}={v}" for k, v in sub.items() if not isinstance(v, (list, dict)))
                if bits:
                    L.append(f"- **{key.replace('_', ' ').title()}:** {bits}")
        L.append("")

    # --- Contrarian ---
    c = r.get("contrarian")
    if c:
        L.append("## Contrarian")
        L.append(f"- **Crowd:** {c.get('crowd_position')} → signal **{c.get('contrarian_signal')}**")
        L += _line("Reasoning", c.get("reasoning"))
        for o in c.get("key_observations") or []:
            L.append(f"  - {o}")
        L.append("")

    # --- Devil's advocate ---
    da = r.get("devil_advocate")
    if da:
        L.append("## Devil's advocate")
        L.append(f"- **Verdict:** {da.get('overall_verdict')}")
        L += _line("Summary", da.get("summary"))
        if da.get("veto_reason"):
            L.append(f"- **VETO:** {da['veto_reason']}")
        for fnd in da.get("findings") or []:
            if isinstance(fnd, dict):
                L.append(f"  - {fnd.get('failure_mode', fnd.get('mode', '?'))}: {fnd.get('description', fnd.get('detail', ''))}")
            else:
                L.append(f"  - {fnd}")
        L += _line("Counter-thesis", da.get("counter_thesis"))
        L.append("")

    errs = r.get("errors") or []
    if errs:
        L.append("## Data notes")
        L += [f"- {e}" for e in errs]
        L.append("")

    return "\n".join(L).rstrip() + "\n"
