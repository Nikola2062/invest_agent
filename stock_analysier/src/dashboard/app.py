"""Streamlit dashboard for the stock analysis system.

Run from project root:
    .venv/bin/streamlit run src/dashboard/app.py

This module is a slim dispatcher. Each view lives under `src/dashboard/views/`
and shared pieces (CSS, helpers, pipeline runner) live in
`src/dashboard/components.py`. (The folder is intentionally NOT named `pages/`
so Streamlit's automatic multipage navigation doesn't run these helper modules
as standalone pages — navigation is the sidebar radio below.)
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import streamlit as st

# Allow running with `streamlit run src/dashboard/app.py` from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config.loader import load_portfolio, reload_all
from src.dashboard.components import (
    configured_symbols,
    init_session_state,
    inject_css,
    list_targets,
    run_selected,
)
from src.dashboard.views import per_ticker, portfolio_fit, post_mortem
from src.pipeline.orchestrator import reset_macro_cache
from src.storage.user_tickers import (
    add_user_ticker,
    detect_market,
    load_user_tickers,
    remove_user_ticker,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")


def _read_mini() -> bool:
    # Read ?mini=1 before set_page_config so we can pick a phone-friendly layout.
    try:
        return str(st.query_params.get("mini", "")).lower() in ("1", "true", "yes")
    except Exception:
        return False


_MINI = _read_mini()
st.set_page_config(
    page_title="Stock Analyser",
    page_icon="📊",
    layout="centered" if _MINI else "wide",
    initial_sidebar_state="collapsed" if _MINI else "expanded",
)

inject_css()
init_session_state()


# ---------- sidebar ----------

with st.sidebar:
    st.title("📊 Stock Analyser")
    st.caption("Long-term + tactical-exit framework")

    page = st.radio(
        "View",
        ["Per-Ticker", "Portfolio Fit", "Post-Mortem"],
        index=0,
    )

    targets = list_targets()
    options = [f"{sym}  ({mkt})" for sym, mkt in targets]
    label_to_pair = {f"{sym}  ({mkt})": (sym, mkt) for sym, mkt in targets}

    selected_label = None
    if page == "Per-Ticker":
        # Only show tickers that have actually been analyzed — selecting an
        # unanalyzed ticker just renders an empty pane, which is confusing.
        analyzed_syms = set(st.session_state["results"].keys())
        analyzed_labels = [lbl for lbl in options if label_to_pair[lbl][0] in analyzed_syms]
        if analyzed_labels:
            selected_label = st.radio(
                "View detail",
                analyzed_labels,
                index=0,
            )
        else:
            st.caption(
                "_No analyses yet. Pick tickers below and press_ **Run pipeline** "
                "_to populate this list._"
            )

    st.divider()

    st.caption("**Pipeline runner**")
    held_symbols = {h.symbol for h in load_portfolio().holdings}
    held_labels = [lbl for lbl in options if label_to_pair[lbl][0] in held_symbols]
    watchlist_labels = [lbl for lbl in options if label_to_pair[lbl][0] not in held_symbols]

    preset = st.radio(
        "Preset",
        ["Held only", "Watchlist only", "All", "Custom"],
        index=0,
        horizontal=False,
    )
    if preset == "Held only":
        default_selection = held_labels
    elif preset == "Watchlist only":
        default_selection = watchlist_labels
    elif preset == "All":
        default_selection = options
    else:
        default_selection = st.session_state.get("custom_selection", held_labels)

    selected_to_run = st.multiselect(
        "Tickers to analyze",
        options=options,
        default=default_selection,
    )
    if preset == "Custom":
        st.session_state["custom_selection"] = selected_to_run

    # ---- Add a ticker on the fly (persisted to data/user_tickers.json) ----
    with st.expander("➕ Add ticker", expanded=False):
        st.caption("Add a symbol that isn't in your portfolio or watchlist config.")
        new_sym_raw = st.text_input(
            "Symbol",
            key="new_ticker_input",
            placeholder="e.g. TSLA or 0700.HK or 700",
        ).strip()
        auto_market = detect_market(new_sym_raw) if new_sym_raw else "US"
        market_choice = st.selectbox(
            "Market",
            options=["US", "HK"],
            index=0 if auto_market == "US" else 1,
            help="Auto-detected from symbol suffix — override if wrong.",
        )
        if st.button("Add", disabled=not new_sym_raw, key="add_ticker_btn"):
            try:
                canonical = add_user_ticker(new_sym_raw, market_choice)
                st.success(f"Added {canonical} ({market_choice}).")
                st.rerun()
            except ValueError as e:
                st.error(str(e))

    # ---- Remove user-added tickers ----
    user_pairs: list[tuple[str, str]] = []
    user_tickers_map = load_user_tickers()
    cfg_syms = configured_symbols()
    for mkt in ("US", "HK"):
        for sym in user_tickers_map.get(mkt, []):
            if sym not in cfg_syms:  # never offer to remove a config-defined symbol
                user_pairs.append((sym, mkt))

    if user_pairs:
        with st.expander(f"🗑️  Remove user-added ({len(user_pairs)})", expanded=False):
            for sym, mkt in user_pairs:
                row_l, row_r = st.columns([3, 1])
                row_l.markdown(f"`{sym}` ({mkt})")
                if row_r.button("×", key=f"rm_{mkt}_{sym}"):
                    remove_user_ticker(sym, mkt)
                    st.session_state["results"].pop(sym, None)
                    st.session_state["errors"].pop(sym, None)
                    st.rerun()

    st.divider()
    col_a, col_b = st.columns(2)
    if col_a.button("Reload configs"):
        reload_all()
        st.success("Configs reloaded.")
    if col_b.button("Refresh macro"):
        reset_macro_cache()
        st.success("Macro cache cleared.")

    run_clicked = st.button(
        f"🔬 Run pipeline ({len(selected_to_run)} ticker{'s' if len(selected_to_run) != 1 else ''})",
        type="primary",
        disabled=not selected_to_run,
    )


# ---------- main ----------

if run_clicked:
    pairs = [label_to_pair[lbl] for lbl in selected_to_run]
    run_selected(pairs)


if page == "Portfolio Fit":
    portfolio_fit.render()
elif page == "Post-Mortem":
    post_mortem.render()
else:
    per_ticker.render(targets, label_to_pair, selected_label)
