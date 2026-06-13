"""Portfolio Fit page — cross-position correlation + concentration analysis."""
from __future__ import annotations

from collections import defaultdict

import streamlit as st

from src.config.loader import load_portfolio
from src.data.fx import convert as fx_convert


def _book_breakdown() -> tuple[float, list[dict]]:
    """Walk current holdings + (if analyzed) current prices and break book down USD.

    Returns (total_book_usd, [{symbol, market, currency, value_usd, weight_pct, sector}]).
    """
    portfolio = load_portfolio()
    results = st.session_state.get("results", {})
    rows: list[dict] = []
    for h in portfolio.holdings:
        if h.shares <= 0:
            continue
        res = results.get(h.symbol)
        price = res.current_price if res is not None else h.cost_basis_per_share
        value_local = h.shares * price
        rows.append({
            "symbol": h.symbol, "market": h.market, "currency": h.currency,
            "value_usd": fx_convert(value_local, h.currency, "USD"),
            "price_basis": "live" if res is not None else "cost",
        })
    total = sum(r["value_usd"] for r in rows) or 1.0
    for r in rows:
        r["weight_pct"] = r["value_usd"] / total * 100
    return total, rows


def _render_concentration_panel() -> None:
    """Deterministic, always-on concentration display — no LLM required."""
    total, rows = _book_breakdown()
    if not rows:
        st.info("No holdings configured in `config/portfolio.yaml` — concentration check skipped.")
        return

    by_name = sorted(rows, key=lambda r: -r["weight_pct"])
    largest = by_name[0]

    by_market: dict[str, float] = defaultdict(float)
    by_currency: dict[str, float] = defaultdict(float)
    for r in rows:
        by_market[r["market"]] += r["weight_pct"]
        by_currency[r["currency"]] += r["weight_pct"]
    top_market = max(by_market.items(), key=lambda kv: kv[1])
    top_ccy = max(by_currency.items(), key=lambda kv: kv[1])

    st.markdown("### 📐 Concentration (deterministic, no LLM)")
    if any(r["price_basis"] == "cost" for r in rows):
        st.caption("_Some positions valued at cost basis — run pipeline for live valuation._")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Book value (USD)", f"${total:,.0f}")
    c2.metric("Largest single name",
              f"{largest['symbol']}",
              f"{largest['weight_pct']:.1f}% of book")
    c3.metric("Largest single market", top_market[0], f"{top_market[1]:.1f}% of book")
    c4.metric("Largest single currency", top_ccy[0], f"{top_ccy[1]:.1f}% of book")

    warnings: list[tuple[str, str]] = []  # (severity, text)
    if largest["weight_pct"] >= 50:
        warnings.append(("error",
            f"🚨 **Extreme single-name concentration** — `{largest['symbol']}` is "
            f"**{largest['weight_pct']:.1f}%** of the book. A single thesis break "
            "or one bad earnings print can permanently impair >50% of capital. "
            "This violates Don't Die — capital preservation."))
    elif largest["weight_pct"] >= 30:
        warnings.append(("warning",
            f"⚠️ Single-name concentration: `{largest['symbol']}` is "
            f"**{largest['weight_pct']:.1f}%** of the book. Above the 30% threshold "
            "typical risk policy considers prudent."))

    if top_market[1] >= 80:
        warnings.append(("error",
            f"🚨 **Single-market concentration** — **{top_market[1]:.1f}%** of the book "
            f"is in `{top_market[0]}`. Regulatory, currency, or geopolitical shock "
            "in this market hits the entire portfolio simultaneously."))
    elif top_market[1] >= 60:
        warnings.append(("warning",
            f"⚠️ Single-market concentration: **{top_market[1]:.1f}%** in `{top_market[0]}`."))

    if top_ccy[1] >= 80 and top_ccy[0] not in ("USD",):
        warnings.append(("warning",
            f"⚠️ Currency concentration: **{top_ccy[1]:.1f}%** of book exposure "
            f"is in `{top_ccy[0]}`. FX moves swing the USD value of the book directly."))

    if warnings:
        for sev, msg in warnings:
            (st.error if sev == "error" else st.warning)(msg)
    else:
        st.success("No single-axis concentration over policy thresholds.")

    with st.expander("Position breakdown", expanded=False):
        st.dataframe(
            [{
                "Symbol": r["symbol"], "Market": r["market"], "Currency": r["currency"],
                "Value (USD)": f"${r['value_usd']:,.0f}",
                "Weight": f"{r['weight_pct']:.1f}%",
                "Priced at": r["price_basis"],
            } for r in by_name],
            hide_index=True, width="stretch",
        )


def render() -> None:
    st.title("🧭 Portfolio Fit")
    st.caption("Cross-position correlation + risk-source concentration analysis (per the investment framework, Ch.8)")

    # Always-on deterministic concentration panel first — visible without needing LLM
    _render_concentration_panel()
    st.divider()

    available_results = list(st.session_state["results"].values())
    if not available_results:
        st.info(
            "💡 Run analysis on at least 2 tickers (Per-Ticker view → Run pipeline) "
            "to unlock the LLM correlation / cluster analysis below."
        )
        return

    if st.button("📊 Generate Portfolio Fit report (LLM)", type="primary"):
        from src.agents.portfolio_fit import analyze_portfolio_fit
        with st.spinner("Computing correlations and analyzing concentration…"):
            try:
                st.session_state["portfolio_fit"] = analyze_portfolio_fit(available_results)
            except Exception as e:
                st.error(f"Portfolio Fit agent failed: {e}")

    pf = st.session_state.get("portfolio_fit")
    if pf is None:
        st.info("Press the button above to run the LLM-driven cluster analysis.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("Positions analyzed", pf.total_positions)
    c2.metric("Total book value", f"${pf.total_book_value_usd:,.0f}")
    c3.metric("Diversification score", f"{pf.diversification_score:.1f}/10")
    st.markdown(f"**Summary:** {pf.summary}")

    if pf.concentration_warnings:
        st.warning("**Concentration warnings:**")
        for w in pf.concentration_warnings:
            st.write(f"• {w}")

    if pf.clusters:
        st.markdown("### Risk clusters")
        for cluster in pf.clusters:
            sev_color = {"low": "#27ae60", "medium": "#e67e22", "high": "#c0392b"}.get(cluster.severity, "#7f8c8d")
            st.markdown(
                f"**<span style='color:{sev_color}'>{cluster.severity.upper()}</span> — {cluster.common_risk_source}**",
                unsafe_allow_html=True,
            )
            st.write(f"Symbols: `{', '.join(cluster.symbols)}`")
            st.write(
                f"Avg pairwise correlation: {cluster.avg_pairwise_correlation:.2f}  |  "
                f"Book concentration: {cluster.concentration_pct_of_book:.1f}%"
            )
            st.markdown("---")

    if pf.diversification_recommendations:
        st.markdown("### Recommendations")
        for r in pf.diversification_recommendations:
            st.write(f"• {r}")
