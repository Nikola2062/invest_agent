"""Per-Ticker page — portfolio overview table + full per-symbol drill-down."""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.config.loader import load_portfolio
from src.dashboard.components import is_mini, level_color, money, signed_pct
from src.models.schemas import AnalysisResult


def render(
    targets: list[tuple[str, str]],
    label_to_pair: dict[str, tuple[str, str]],
    selected_label: str | None,
) -> None:
    _render_overview(targets)
    st.divider()

    # In mini mode the sidebar is collapsed, so offer ticker selection in the
    # main pane instead of relying on the sidebar radio.
    if is_mini():
        analyzed_labels = [lbl for lbl in label_to_pair if label_to_pair[lbl][0] in st.session_state["results"]]
        if not analyzed_labels:
            st.info("No analyses yet. Open the sidebar (tap ›) and run the pipeline.")
            return
        idx = analyzed_labels.index(selected_label) if selected_label in analyzed_labels else 0
        selected_label = st.selectbox("View detail", analyzed_labels, index=idx)

    if not selected_label:
        return

    selected_sym, _ = label_to_pair[selected_label]
    res: AnalysisResult | None = st.session_state["results"].get(selected_sym)
    err = st.session_state["errors"].get(selected_sym)

    if err:
        st.error(f"Analysis error for {selected_sym}: {err}")
        if "DeepSeek" in err or "api_key" in err.lower():
            st.info("Add your DeepSeek API key to `config/secrets.yaml` under `deepseek.api_key` and rerun.")
        return

    if res is None:
        st.info(f"Press **Run pipeline** in the sidebar to analyze {selected_sym}.")
        return

    _render_detail(selected_sym, res)


# ---------- overview table ----------

def _render_overview(targets: list[tuple[str, str]]) -> None:
    st.title("Portfolio Overview")
    portfolio = load_portfolio()

    rows = []
    for sym, mkt in targets:
        res: AnalysisResult | None = st.session_state["results"].get(sym)
        holding = portfolio.find(sym)
        shares_str = f"{holding.shares:.0f}" if holding else ""
        cost_str = money(holding.cost_basis_per_share, holding.currency) if holding else ""
        if res is None:
            rows.append({
                "Symbol": sym, "Market": mkt,
                "Held": "✓" if holding else "", "Shares": shares_str, "Cost basis": cost_str,
                "Current": "—", "P&L %": "—", "Quality": "—", "MoS %": "—",
                "P(dd≥15%)": "—", "Tech": "—", "Action": "Not yet analyzed",
            })
            continue
        pnl_pct = (res.current_price / holding.cost_basis_per_share - 1) * 100 if holding else None
        action_label = res.if_held.tactical.label or "HOLD" if holding else res.if_not_held.recommendation
        da_verdict = "—"
        if res.devil_advocate:
            da_verdict = {"pass": "✅", "pass_with_concerns": "⚠️", "veto": "🚫"}.get(
                res.devil_advocate.overall_verdict, "—"
            )
        tech_emoji = "—"
        if res.technical:
            tech_emoji = {
                "strong_bullish": "🟢🟢", "bullish": "🟢", "neutral": "⚪",
                "bearish": "🔴", "strong_bearish": "🔴🔴",
            }.get(res.technical.composite_signal, "—")
        rows.append({
            "Symbol": sym, "Market": mkt,
            "Held": "✓" if holding else "", "Shares": shares_str, "Cost basis": cost_str,
            "Current": money(res.current_price, res.currency),
            "P&L %": signed_pct(pnl_pct) if pnl_pct is not None else "—",
            "Quality": f"{res.fundamental.quality_score:.1f}/10",
            "MoS %": signed_pct(res.valuation.margin_of_safety_pct),
            "P(dd≥15%)": f"{res.risk.drawdown_probabilities.get('15', 0) * 100:.0f}%",
            "Tech": tech_emoji,
            "Action": action_label, "DA": da_verdict,
        })

    if not rows:
        st.info("No tickers configured. Edit config/portfolio.yaml and config/universe.yaml.")
        return

    # Mini: one card per ticker (a 13-column table is unreadable on a phone).
    if is_mini():
        for r in rows:
            with st.container(border=True):
                held = " · held" if r["Held"] else ""
                st.markdown(f"**{r['Symbol']}** ({r['Market']}{held}) — **{r['Action']}**")
                st.caption(
                    f"P&L {r['P&L %']} · MoS {r['MoS %']} · Q {r['Quality']} · "
                    f"P(dd≥15%) {r['P(dd≥15%)']} · Tech {r['Tech']} · DA {r.get('DA', '—')}"
                )
        return

    column_config = {
        "Symbol":     st.column_config.TextColumn("Symbol", width="small"),
        "Market":     st.column_config.TextColumn("Mkt", width="small"),
        "Held":       st.column_config.TextColumn("Held", width="small"),
        "Shares":     st.column_config.TextColumn("Shares", width="small"),
        "Cost basis": st.column_config.TextColumn("Cost basis", width="medium"),
        "Current":    st.column_config.TextColumn("Current", width="medium"),
        "P&L %":      st.column_config.TextColumn("P&L %", width="small"),
        "Quality":    st.column_config.TextColumn("Quality", width="small"),
        "MoS %":      st.column_config.TextColumn("MoS %", width="small"),
        "P(dd≥15%)":  st.column_config.TextColumn("P(dd≥15%)", width="small"),
        "Tech":       st.column_config.TextColumn("Tech", width="small"),
        "Action":     st.column_config.TextColumn("Action", width="medium"),
        "DA":         st.column_config.TextColumn("DA", width="small"),
    }
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config=column_config)


# ---------- per-ticker detail ----------

def _render_detail(selected_sym: str, res: AnalysisResult) -> None:
    holding = res.position
    st.header(f"{selected_sym} — {res.fundamental.thesis_one_liner}")

    _render_competence_badge(res)
    _render_devil_banner(res)
    _render_top_metrics(res, holding)
    _render_action_panels(res)

    st.divider()
    _render_tabs(selected_sym, res)


def _render_competence_badge(res: AnalysisResult) -> None:
    if res.competence is None:
        return
    badge = {
        "in_circle": ("✅", "#27ae60", "In circle of competence"),
        "borderline": ("🟡", "#e67e22", "Borderline — outside declared circle"),
        "out_of_circle": ("🚫", "#c0392b", "OUTSIDE circle of competence"),
    }.get(res.competence.verdict, ("?", "#7f8c8d", "unknown"))
    emoji, color, label = badge
    st.markdown(
        f"<small>{emoji} <span style='color:{color}'>**{label}**</span> — _{res.competence.reasoning}_</small>",
        unsafe_allow_html=True,
    )
    with st.expander("Why this competence verdict? How do I override it?", expanded=False):
        from src.config.loader import load_competence
        cfg = load_competence()
        st.markdown(f"**Verdict:** `{res.competence.verdict}`")
        if res.competence.matched_categories:
            st.markdown(f"**Matched categories:** `{', '.join(res.competence.matched_categories)}`")
        st.markdown(
            f"**Source:** `config/competence.yaml` — current policy: "
            f"`{cfg.on_out_of_circle.get('policy', 'analyze_but_flag')}`"
        )
        st.markdown(
            """
**How to override:**

1. **Allow this specific symbol** — add the symbol to `always_in_circle` in `config/competence.yaml`:
   ```yaml
   always_in_circle:
     - BRK.B
     - """ + res.symbol + """
   ```

2. **Loosen the keyword filter** — remove the offending keyword from `out_of_circle_keywords` if you want to assess this kind of business going forward.

3. **Skip such tickers entirely** — set `on_out_of_circle.policy: skip` to stop running the full pipeline on out-of-circle names (saves LLM cost).

After editing, click **Reload configs** in the sidebar.
            """
        )
        st.caption(
            "Per the investment framework, Ch.12 (能力圈): "
            "the largest risk is not knowing what you don't know — be honest with this list."
        )


def _render_devil_banner(res: AnalysisResult) -> None:
    if res.devil_advocate is None:
        return
    da = res.devil_advocate
    if da.overall_verdict == "veto":
        st.error(f"🚫 **Devil's Advocate VETO** — {da.veto_reason or da.summary}")
    elif da.overall_verdict == "pass_with_concerns":
        st.warning(f"⚠️ **Devil's Advocate flags concerns** — {da.summary}")
    else:
        st.success(f"✅ **Devil's Advocate: pass** — {da.summary}")


def _render_top_metrics(res: AnalysisResult, holding) -> None:
    metrics: list[tuple[str, str, str | None]] = [
        ("Current price", money(res.current_price, res.currency), None),
    ]
    if holding:
        pnl_pct = (res.current_price / holding.cost_basis_per_share - 1) * 100
        metrics.append(("Position", f"{holding.shares:.0f} sh", f"P&L {signed_pct(pnl_pct)} on cost"))
    else:
        metrics.append(("Position", "not held", None))
    metrics.append(("Quality", f"{res.fundamental.quality_score:.1f}/10", res.fundamental.moat_strength))
    metrics.append(("Intrinsic (base)",
                    money(res.valuation.intrinsic_base, res.valuation.currency),
                    signed_pct(res.valuation.margin_of_safety_pct)))
    metrics.append(("Realized vol", f"{res.risk.realized_vol_annualized_pct:.1f}%",
                    f"P(dd≥20%) {res.risk.drawdown_probabilities.get('20', 0) * 100:.0f}%"))

    per_row = 2 if is_mini() else 5  # 5-up on desktop, 2-up stacked on phone
    for i in range(0, len(metrics), per_row):
        cols = st.columns(per_row)
        for col, (label, value, delta) in zip(cols, metrics[i:i + per_row]):
            col.metric(label, value, delta) if delta is not None else col.metric(label, value)


def _render_action_panels(res: AnalysisResult) -> None:
    # Stack the two panels on a phone; side-by-side on desktop.
    left, right = (st.container(), st.container()) if is_mini() else st.columns(2)

    with left:
        panel_color = level_color(res.if_held.tactical.label)
        st.markdown(
            f"### IF HELD — <span style='color:{panel_color}'>{res.if_held.tactical.label or 'NO ACTION'}</span>",
            unsafe_allow_html=True,
        )
        st.write(res.if_held.tactical.rationale)
        if res.if_held.tactical.tax_notes:
            with st.expander("Tax notes", expanded=False):
                for n in res.if_held.tactical.tax_notes:
                    st.write(f"• {n}")
        if res.if_held.immediate_orders:
            st.markdown("**Immediate orders:**")
            for o in res.if_held.immediate_orders:
                st.code(
                    f"{o.side} {o.quantity} {o.symbol} {o.order_type} "
                    f"@ {o.limit_price if o.limit_price else 'MKT'} {o.time_in_force}",
                    language="text",
                )
                st.caption(o.rationale)
        if res.if_held.rebuy_orders:
            st.markdown("**Pre-committed rebuy orders (conditional):**")
            for o in res.if_held.rebuy_orders:
                st.code(
                    f"{o.side} {o.quantity} {o.symbol} {o.order_type} "
                    f"@ {o.limit_price} {o.time_in_force} [conditional]",
                    language="text",
                )
                st.caption(o.rationale)
        if res.if_held.tactical.hedge_recommended:
            st.info("Hedge recommended for remaining position. See the 🛡️ Hedge tab.")
        if not res.if_held.immediate_orders and not res.if_held.rebuy_orders:
            st.success("No immediate action. Continue to hold.")

    with right:
        rec = res.if_not_held.recommendation
        rec_color = {"BUY_NOW": "#27ae60", "WAIT_FOR_PRICE": "#3498db", "PASS": "#7f8c8d"}.get(rec, "#7f8c8d")
        st.markdown(
            f"### IF NOT HELD — <span style='color:{rec_color}'>{rec}</span>",
            unsafe_allow_html=True,
        )
        st.write(res.if_not_held.rationale)
        if res.if_not_held.entry_orders:
            st.markdown("**Entry orders:**")
            for o in res.if_not_held.entry_orders:
                st.code(
                    f"{o.side} {o.quantity} {o.symbol} {o.order_type} "
                    f"@ {o.limit_price} {o.time_in_force}"
                    + (" [conditional]" if o.conditional else ""),
                    language="text",
                )
                st.caption(o.rationale)


def _render_tabs(selected_sym: str, res: AnalysisResult) -> None:
    sections = [
        ("📋 Fundamental", lambda: _render_fundamental_tab(res)),
        ("💰 Valuation", lambda: _render_valuation_tab(selected_sym, res)),
        ("📑 Financials", lambda: _render_financials_tab(res)),
        ("⚠️ Risk", lambda: _render_risk_tab(res)),
        ("📊 Technical", lambda: _render_technical_tab(res)),
        ("📈 Scenarios", lambda: _render_scenarios_tab(res)),
        ("🔮 Catalysts", lambda: _render_catalysts_tab(res)),
        ("🧨 Contrarian", lambda: _render_contrarian_tab(res)),
        ("🛡️ Hedge", lambda: _render_hedge_tab(res)),
        ("🔥 Devil's Advocate", lambda: _render_devil_tab(res)),
        ("🌍 Macro", lambda: _render_macro_tab()),
    ]
    if is_mini():
        # A scrollable strip of 11 tabs is unusable on a phone — pick via dropdown.
        choice = st.selectbox("Section", [label for label, _ in sections], key=f"section_{selected_sym}")
        dict(sections)[choice]()
    else:
        tabs = st.tabs([label for label, _ in sections])
        for tab, (_, fn) in zip(tabs, sections):
            with tab:
                fn()


def _fmt_big(v: float | None) -> str:
    if v is None:
        return "—"
    a = abs(v)
    if a >= 1e12:
        return f"{v/1e12:,.2f}T"
    if a >= 1e9:
        return f"{v/1e9:,.2f}B"
    if a >= 1e6:
        return f"{v/1e6:,.2f}M"
    return f"{v:,.0f}"


def _fmt_delta(curr: float | None, prev: float | None) -> str:
    if curr is None or prev is None or prev == 0:
        return "—"
    pct = (curr - prev) / abs(prev) * 100
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _statement_df(lines, periods: list[str]) -> pd.DataFrame:
    rows = []
    for ln in lines:
        row = {"Item": ln.label}
        for i, p in enumerate(periods):
            row[p] = _fmt_big(ln.values[i]) if i < len(ln.values) else "—"
        if len(ln.values) >= 2:
            row["Δ vs prior"] = _fmt_delta(ln.values[0], ln.values[1])
        else:
            row["Δ vs prior"] = "—"
        rows.append(row)
    return pd.DataFrame(rows)


def _render_financials_tab(res: AnalysisResult) -> None:
    fr = res.financial_report
    if fr is None:
        st.info(
            "Financial Report did not run for this ticker. Re-run the pipeline; if the "
            "fetch fails repeatedly the underlying ticker may not publish statements via yfinance."
        )
        return

    st.caption(f"Reporting currency: **{fr.currency}**")
    if fr.fetch_notes:
        with st.expander("Data-source notes", expanded=False):
            for n in fr.fetch_notes:
                st.write(f"• {n}")

    # ---- Period tables (annual + quarterly) ----
    if fr.annual and fr.annual.periods:
        st.markdown(f"#### 📆 Annual — last {len(fr.annual.periods)} period(s)")
        sub1, sub2, sub3 = st.tabs(["Income", "Balance Sheet", "Cash Flow"])
        with sub1:
            if fr.annual.income:
                st.dataframe(_statement_df(fr.annual.income, fr.annual.periods),
                             hide_index=True, width="stretch")
            else:
                st.caption("No annual income-statement rows available.")
        with sub2:
            if fr.annual.balance:
                st.dataframe(_statement_df(fr.annual.balance, fr.annual.periods),
                             hide_index=True, width="stretch")
            else:
                st.caption("No annual balance-sheet rows available.")
        with sub3:
            if fr.annual.cashflow:
                st.dataframe(_statement_df(fr.annual.cashflow, fr.annual.periods),
                             hide_index=True, width="stretch")
            else:
                st.caption("No annual cash-flow rows available.")
    else:
        st.warning("No annual statements were retrieved.")

    if fr.quarterly and fr.quarterly.periods:
        st.markdown(f"#### 🗓️ Quarterly — last {len(fr.quarterly.periods)} period(s)")
        sub1, sub2, sub3 = st.tabs(["Income (Q)", "Balance Sheet (Q)", "Cash Flow (Q)"])
        with sub1:
            if fr.quarterly.income:
                st.dataframe(_statement_df(fr.quarterly.income, fr.quarterly.periods),
                             hide_index=True, width="stretch")
            else:
                st.caption("No quarterly income-statement rows available.")
        with sub2:
            if fr.quarterly.balance:
                st.dataframe(_statement_df(fr.quarterly.balance, fr.quarterly.periods),
                             hide_index=True, width="stretch")
            else:
                st.caption("No quarterly balance-sheet rows available.")
        with sub3:
            if fr.quarterly.cashflow:
                st.dataframe(_statement_df(fr.quarterly.cashflow, fr.quarterly.periods),
                             hide_index=True, width="stretch")
            else:
                st.caption("No quarterly cash-flow rows available.")
    else:
        st.warning("No quarterly statements were retrieved.")

    # ---- LLM deep resolution ----
    st.markdown("---")
    st.markdown("### 🧠 Deep resolution")
    dr = fr.deep_resolution
    if dr is None:
        st.info(
            "Deep-resolution LLM analysis is not available for this run "
            "(LLM call failed or was skipped). The raw statements above are still usable."
        )
        return

    st.success(f"**Summary:** {dr.summary}")
    st.markdown("**📈 Revenue trend**")
    st.write(dr.revenue_trend)
    st.markdown("**🧮 Margin trend**")
    st.write(dr.margin_trend)
    st.markdown("**🏦 Balance-sheet trend**")
    st.write(dr.balance_sheet_trend)
    st.markdown("**💵 Cash-flow quality**")
    st.write(dr.cash_flow_quality)
    st.markdown("**🎯 Capital allocation observed**")
    st.write(dr.capital_allocation_observed)

    col_pos, col_neg = st.columns(2)
    with col_pos:
        st.markdown("**✅ Key positives**")
        if dr.key_positives:
            for p in dr.key_positives:
                st.write(f"• {p}")
        else:
            st.caption("_(none called out)_")
    with col_neg:
        st.markdown("**🚩 Key red flags**")
        if dr.key_red_flags:
            for r in dr.key_red_flags:
                st.write(f"• {r}")
        else:
            st.caption("_(none called out)_")

    st.markdown("---")
    st.markdown("### 💡 Investment implication")
    st.info(dr.investment_implication)


def _render_fundamental_tab(res: AnalysisResult) -> None:
    f = res.fundamental
    cc1, cc2, cc3 = st.columns(3)
    cc1.metric("Quality score", f"{f.quality_score:.1f}/10")
    cc2.metric("Moat", f.moat_strength)
    cc3.metric("Balance sheet", f.balance_sheet_health)
    st.markdown("**Thesis:** " + f.thesis_one_liner)
    st.markdown(f"**Moat:** {f.moat_assessment}")
    st.markdown(f"**Growth outlook:** {f.growth_outlook}")
    st.markdown(f"**Capital allocation:** {f.capital_allocation}")
    if f.red_flags:
        st.markdown("**Red flags:**")
        for r in f.red_flags:
            st.write(f"• {r}")


def _render_valuation_tab(selected_sym: str, res: AnalysisResult) -> None:
    v = res.valuation
    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Range", x=[selected_sym],
        y=[v.intrinsic_high - v.intrinsic_low], base=[v.intrinsic_low],
        marker_color="rgba(46,204,113,0.4)", showlegend=False,
    ))
    fig.add_hline(
        y=v.intrinsic_base, line_dash="dash", line_color="green",
        annotation_text=f"Intrinsic base {money(v.intrinsic_base, v.currency)}",
    )
    fig.add_hline(
        y=res.current_price, line_color="red",
        annotation_text=f"Current {money(res.current_price, res.currency)}",
    )
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=20, b=20), yaxis_title=f"Price ({v.currency})")
    st.plotly_chart(fig, width="stretch")

    vc1, vc2, vc3 = st.columns(3)
    vc1.metric("Margin of safety", signed_pct(v.margin_of_safety_pct))
    vc2.metric("DCF value", money(v.dcf_value, v.currency) if v.dcf_value else "n/a")
    vc3.metric("Multiples value", money(v.multiples_value, v.currency) if v.multiples_value else "n/a")
    st.markdown(f"**Methodology:** {v.methodology_notes}")
    st.caption(f"Confidence: {v.confidence}")


def _render_risk_tab(res: AnalysisResult) -> None:
    r = res.risk
    keys = ["10", "15", "20", "25"]
    probs = [r.drawdown_probabilities.get(k, 0) * 100 for k in keys]
    fig = go.Figure(data=[go.Bar(
        x=[f"≥ {k}%" for k in keys], y=probs,
        marker_color=["#f1c40f", "#e67e22", "#e74c3c", "#1a1a1a"],
    )])
    fig.update_layout(
        height=300, yaxis_title="Probability (%)",
        title=f"P(drawdown ≥ X%) over {r.horizon_days}d horizon",
        margin=dict(l=20, r=20, t=40, b=20),
    )
    st.plotly_chart(fig, width="stretch")

    st.markdown("**Scenarios:**")
    for s in r.scenarios:
        st.markdown(
            f"- **{s.name}** ({s.probability * 100:.0f}%): return {signed_pct(s.expected_return_pct)}, "
            f"drawdown {s.expected_drawdown_pct:.1f}% — {s.rationale}"
        )

    if r.key_macro_signals:
        st.markdown("**Key signals driving this assessment:**")
        for s in r.key_macro_signals:
            st.write(f"• {s}")


def _render_scenarios_tab(res: AnalysisResult) -> None:
    fs = res.forward_scenarios
    if fs is None or not fs.scenarios:
        st.info("Forward Scenarios agent did not run, or produced no scenarios.")
        return
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Current", money(fs.current_price, fs.currency))
    sc2.metric("Prob-weighted target", money(fs.probability_weighted_target, fs.currency))
    sc3.metric("Expected return", signed_pct(fs.expected_return_pct))
    st.markdown(f"**Summary:** {fs.summary}")
    st.markdown(f"**Horizon:** {fs.horizon_days} days")
    scen_fig = go.Figure()
    for s in fs.scenarios:
        scen_fig.add_trace(go.Bar(
            name=f"{s.name} ({s.probability * 100:.0f}%)",
            x=[s.name],
            y=[s.target_price_high - s.target_price_low], base=[s.target_price_low],
        ))
    scen_fig.add_hline(
        y=fs.current_price, line_color="red",
        annotation_text=f"current {fs.current_price:.2f}",
    )
    scen_fig.update_layout(
        height=320, yaxis_title=f"Price ({fs.currency})", barmode="overlay",
        margin=dict(l=20, r=20, t=20, b=20),
    )
    st.plotly_chart(scen_fig, width="stretch")
    st.markdown("**Scenario details:**")
    for s in fs.scenarios:
        with st.expander(
            f"{s.name} — {s.probability * 100:.0f}% probability — return {signed_pct(s.return_pct_base)}"
        ):
            st.markdown(
                f"**Target price range:** {money(s.target_price_low, fs.currency)} – "
                f"**{money(s.target_price_base, fs.currency)}** – {money(s.target_price_high, fs.currency)}"
            )
            st.markdown(f"**Estimated max drawdown during path:** {s.drawdown_pct_estimated:+.1f}%")
            st.markdown(f"**Key drivers:** {', '.join(s.key_drivers) or '—'}")
            st.write(s.rationale)


def _render_catalysts_tab(res: AnalysisResult) -> None:
    fc = res.forward_catalysts
    if fc is None:
        st.info("No forward catalysts available (Information Retrieval agent did not run, or returned empty).")
        return
    st.markdown(f"**Sentiment:** {signed_pct(fc.sentiment_score * 100)} — {fc.sentiment_summary}")
    if fc.macro_overlay:
        st.markdown("**Macro overlay relevant to this name:**")
        for o in fc.macro_overlay:
            st.write(f"• {o}")
    if fc.key_catalysts:
        st.markdown(f"**Key catalysts (next {fc.horizon_days}d):**")
        cat_rows = []
        for c in fc.key_catalysts:
            arrow = {"positive": "↑", "negative": "↓", "uncertain": "?"}.get(c.direction, "?")
            cat_rows.append({
                "Date": c.expected_date.isoformat() if c.expected_date else "—",
                "Dir": arrow,
                "Event": c.event,
                "Magnitude": f"~{c.expected_magnitude_pct:.1f}%" if c.expected_magnitude_pct else "—",
                "Conf": c.confidence,
                "Rationale": c.rationale,
            })
        st.dataframe(pd.DataFrame(cat_rows), hide_index=True, width="stretch")


def _render_contrarian_tab(res: AnalysisResult) -> None:
    ca = res.contrarian
    if ca is None:
        st.info("Contrarian agent did not run (data may be unavailable for non-US tickers on free-tier sources).")
        return
    crowd_color = {
        "euphoric": "#c0392b", "bullish": "#e67e22", "neutral": "#7f8c8d",
        "bearish": "#3498db", "despondent": "#27ae60",
    }.get(ca.crowd_position, "#7f8c8d")
    signal_color = {
        "strong_buy": "#27ae60", "buy": "#2ecc71", "neutral": "#7f8c8d",
        "pass": "#e67e22", "strong_pass": "#c0392b",
    }.get(ca.contrarian_signal, "#7f8c8d")
    cc1, cc2, cc3 = st.columns(3)
    cc1.markdown(
        f"**Crowd position:** <span style='color:{crowd_color}'>**{ca.crowd_position.upper()}**</span>",
        unsafe_allow_html=True,
    )
    cc2.markdown(
        f"**Contrarian signal:** <span style='color:{signal_color}'>**{ca.contrarian_signal.upper()}**</span>",
        unsafe_allow_html=True,
    )
    cc3.markdown(f"**Data quality:** {ca.data_quality}")
    st.markdown(f"**Reasoning:** {ca.reasoning}")
    if ca.key_observations:
        st.markdown("**Key observations:**")
        for obs in ca.key_observations:
            st.write(f"• {obs}")


def _render_hedge_tab(res: AnalysisResult) -> None:
    hp = res.hedge_plan
    if hp is None:
        st.info(
            "No hedge plan generated. Hedging only runs when Tactical Exit recommends it "
            "(RED_DEFENSIVE or higher with hedge_remainder=true) AND the position is above "
            "the configured minimum hedge size."
        )
        return
    if not hp.candidates:
        st.warning("Hedging agent ran but found no suitable candidates for this position.")
        return
    st.markdown(f"**Position to hedge:** {hp.symbol_being_hedged} (~${hp.position_value_usd:,.0f})")
    st.markdown(f"**Rationale:** {hp.rationale}")
    rec_idx = hp.recommended_index
    for i, c in enumerate(hp.candidates):
        label = "✅ RECOMMENDED" if i == rec_idx else f"#{i + 1}"
        with st.expander(f"{label} — `{c.instrument}` ({c.instrument_kind})", expanded=(i == rec_idx)):
            cc1, cc2 = st.columns(2)
            cc1.metric("90d correlation", f"{c.correlation_90d:+.2f}" if c.correlation_90d is not None else "n/a")
            cc2.metric(
                "Suggested notional",
                f"${c.suggested_notional_usd:,.0f}" if c.suggested_notional_usd else "n/a",
            )
            st.write(c.rationale)
            if c.contract_specs:
                st.caption(c.contract_specs)
    if hp.notes:
        st.markdown("**Notes:**")
        for n in hp.notes:
            st.write(f"• {n}")


def _render_devil_tab(res: AnalysisResult) -> None:
    da = res.devil_advocate
    if da is None:
        st.info("Devil's Advocate did not run (skip flag set, or agent failed). See errors below if any.")
        return
    verdict_color = {
        "pass": "#27ae60", "pass_with_concerns": "#e67e22", "veto": "#c0392b",
    }.get(da.overall_verdict, "#7f8c8d")
    st.markdown(
        f"### Verdict: <span style='color:{verdict_color}'>**{da.overall_verdict.upper()}**</span>",
        unsafe_allow_html=True,
    )
    st.markdown(f"_{da.summary}_")
    if da.veto_reason:
        st.error(f"**Veto reason:** {da.veto_reason}")
    st.markdown("---")
    st.markdown("**Counter-thesis (the strongest bear case):**")
    st.write(da.counter_thesis)
    if da.findings:
        st.markdown("---")
        st.markdown(f"**Findings ({len(da.findings)})**")
        for finding in da.findings:
            severity_emoji = {"info": "ℹ️", "concern": "⚠️", "veto": "🚫"}.get(finding.severity, "•")
            with st.expander(
                f"{severity_emoji} [{finding.severity.upper()}] {finding.category} — {finding.finding}",
                expanded=(finding.severity in ("veto", "concern")),
            ):
                st.markdown(f"**Evidence:** {finding.evidence}")
                st.markdown(f"**Recommendation:** {finding.recommendation}")


def _render_technical_tab(res: AnalysisResult) -> None:
    t = res.technical
    if t is None:
        st.info("Technical Division did not run for this ticker (insufficient history or upstream error).")
        return

    # Composite banner colored by polarity
    banner_color = {
        "strong_bullish": "#27ae60", "bullish": "#2ecc71", "neutral": "#7f8c8d",
        "bearish": "#e67e22", "strong_bearish": "#c0392b",
    }.get(t.composite_signal, "#7f8c8d")
    st.markdown(
        f"### Composite: <span style='color:{banner_color}'>**{t.composite_signal.upper()}**</span>",
        unsafe_allow_html=True,
    )
    st.caption(t.composite_rationale)
    st.markdown("---")

    # ---- Structure ----
    st.markdown("#### 📐 Structure")
    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("Trend", t.structure.trend.replace("_", " "))
    sc2.metric("Stage", t.structure.stage)
    sc3.metric("Confidence", f"{t.structure.confidence:.0%}")
    st.caption(t.structure.structure_summary)
    if t.structure.pivots:
        # Plot pivots
        pivot_df = pd.DataFrame([
            {"date": p.date, "price": p.price, "kind": p.kind}
            for p in t.structure.pivots
        ])
        fig = go.Figure()
        for kind, color in [("HH", "#27ae60"), ("HL", "#2ecc71"),
                            ("LH", "#e67e22"), ("LL", "#c0392b")]:
            sub = pivot_df[pivot_df["kind"] == kind]
            if not sub.empty:
                fig.add_trace(go.Scatter(
                    x=sub["date"], y=sub["price"], mode="markers+text",
                    marker=dict(color=color, size=10),
                    text=sub["kind"], textposition="top center", name=kind,
                ))
        fig.update_layout(height=260, margin=dict(l=20, r=20, t=20, b=20),
                          yaxis_title="Price", showlegend=True)
        st.plotly_chart(fig, width="stretch")
    st.markdown("---")

    # ---- Volume ----
    st.markdown("#### 📊 Volume / Flow")
    vc1, vc2, vc3, vc4 = st.columns(4)
    flow_color = {"accumulation": "#27ae60", "neutral": "#7f8c8d", "distribution": "#c0392b"}.get(
        t.volume.institutional_flow, "#7f8c8d",
    )
    vc1.markdown(
        f"**Flow:** <span style='color:{flow_color}'>**{t.volume.institutional_flow.upper()}**</span>",
        unsafe_allow_html=True,
    )
    vc2.metric("OBV trend", t.volume.obv_trend)
    vc3.metric("20d vs 50d vol", f"{t.volume.volume_expansion_pct:+.1f}%")
    vc4.metric("UDV / DDV", f"{t.volume.up_down_volume_ratio:.2f}")
    if t.volume.last_earnings_volume_spike_x:
        st.caption(f"Last earnings volume spike: {t.volume.last_earnings_volume_spike_x:.1f}× baseline")
    if t.volume.signals:
        st.markdown("**Signals:** " + ", ".join(t.volume.signals))
    st.markdown("---")

    # ---- Cost basis ----
    st.markdown("#### 🏷️ Cost-basis distribution")
    cbc1, cbc2 = st.columns(2)
    cbc1.metric("Trapped supply above", f"{t.cost_basis.trapped_supply_pct:.0f}%")
    cbc2.metric("Accumulation below", f"{t.cost_basis.accumulation_pct:.0f}%")
    if t.cost_basis.hvn_levels:
        hvn_df = pd.DataFrame([{
            "Range": f"${lv.price_low:.2f} – ${lv.price_high:.2f}",
            "% of volume": f"{lv.volume_pct_of_window:.1f}%",
            "Position": lv.position_vs_current,
            "Role": lv.role,
        } for lv in t.cost_basis.hvn_levels])
        st.dataframe(hvn_df, hide_index=True, width="stretch")
    st.caption(t.cost_basis.summary)
    st.markdown("---")

    # ---- Relative strength ----
    st.markdown("#### 🥊 Relative strength")
    rs_color = {
        "strong_leader": "#27ae60", "leader": "#2ecc71", "neutral": "#7f8c8d",
        "laggard": "#e67e22", "weak_laggard": "#c0392b",
    }.get(t.relative_strength.signal, "#7f8c8d")
    st.markdown(
        f"**Signal:** <span style='color:{rs_color}'>**{t.relative_strength.signal.upper()}**</span>"
        + (f"  ·  vs `{t.relative_strength.benchmark_sector_etf}` (sector) "
           f"and `{t.relative_strength.benchmark_index}` (index)"
           if t.relative_strength.benchmark_sector_etf else ""),
        unsafe_allow_html=True,
    )
    rs_rows = []
    for label, key in [
        ("vs Sector 90d", "vs_sector_etf_90d"), ("vs Sector 365d", "vs_sector_etf_365d"),
        ("vs Index 90d",  "vs_index_90d"),      ("vs Index 365d",  "vs_index_365d"),
    ]:
        val = getattr(t.relative_strength, key)
        rs_rows.append({"Window": label, "Ratio": f"{val:.2f}" if val is not None else "—"})
    st.dataframe(pd.DataFrame(rs_rows), hide_index=True, width="stretch")
    st.markdown("---")

    # ---- Price map ----
    st.markdown("#### 🗺️ Price Map")
    pmc1, pmc2 = st.columns(2)
    pmc1.metric("Key support", f"${t.price_map.key_support:.2f}" if t.price_map.key_support else "—")
    pmc2.metric("Key resistance", f"${t.price_map.key_resistance:.2f}" if t.price_map.key_resistance else "—")
    if t.price_map.zones:
        zone_rows = []
        for i, z in enumerate(t.price_map.zones):
            marker = " ← CURRENT" if i == t.price_map.current_zone_index else ""
            zone_rows.append({
                "Range": f"${z.price_low:.2f} – ${z.price_high:.2f}{marker}",
                "Label": z.label, "Rationale": z.rationale,
            })
        st.dataframe(pd.DataFrame(zone_rows), hide_index=True, width="stretch")
    st.caption(t.price_map.summary)


def _render_macro_tab() -> None:
    from src.data.macro import fetch_macro
    m = fetch_macro()
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("10Y-3M spread", f"{m.yield_curve_10y3m_pct:.2f}%" if m.yield_curve_10y3m_pct is not None else "n/a")
    mc2.metric("VIX", f"{m.vix_level:.1f}" if m.vix_level is not None else "n/a")
    mc3.metric("S&P vs 52w high", f"{m.sp500_drawdown_pct:.1f}%" if m.sp500_drawdown_pct is not None else "n/a")
    mc4.metric("HSI vs 52w high", f"{m.hsi_drawdown_pct:.1f}%" if m.hsi_drawdown_pct is not None else "n/a")
    if m.signals:
        st.warning("Active macro signals:")
        for s in m.signals:
            st.write(f"• {s}")
    else:
        st.success("No macro stress signals firing.")
