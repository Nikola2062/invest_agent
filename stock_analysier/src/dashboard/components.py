"""Shared dashboard pieces: CSS, helpers, session-state init, pipeline progress runner.

Split out of `app.py` so per-page modules can stay focused on their own UI.
"""
from __future__ import annotations

import logging
import time

import streamlit as st

from src.config.loader import load_portfolio, load_universe
from src.llm.client import LLMConfigError
from src.pipeline.orchestrator import PIPELINE_STEPS, analyze
from src.storage.user_tickers import load_user_tickers


_CUSTOM_CSS = """
<style>
/* Hide Streamlit chrome for a cleaner look */
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* Typography */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Inter", "Segoe UI",
                 "SF Pro Display", "Helvetica Neue", Arial, sans-serif;
    letter-spacing: -0.005em;
}
h1, h2, h3, h4 {
    letter-spacing: -0.025em !important;
    font-weight: 600 !important;
}
h1 { font-size: 1.9rem !important; }
h2 { font-size: 1.4rem !important; }
h3 { font-size: 1.15rem !important; }

/* Metric cards — soft container instead of plain text */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.03);
    padding: 14px 18px;
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.06);
    transition: border-color 0.15s ease;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(61, 184, 255, 0.3);
}
[data-testid="stMetricLabel"] {
    font-size: 0.78rem !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    opacity: 0.7;
}
[data-testid="stMetricValue"] {
    font-size: 1.35rem !important;
    font-weight: 600 !important;
}

/* Status containers — rounder, softer */
[data-testid="stStatus"] {
    border-radius: 12px !important;
    border: 1px solid rgba(255, 255, 255, 0.07) !important;
}

/* Buttons */
.stButton > button {
    border-radius: 10px;
    font-weight: 500;
    transition: transform 0.05s ease, box-shadow 0.15s ease;
}
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #3DB8FF 0%, #2196F3 100%);
    border: none;
    box-shadow: 0 4px 14px rgba(33, 150, 243, 0.25);
}
.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(33, 150, 243, 0.35);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #161B22 100%);
    border-right: 1px solid rgba(255, 255, 255, 0.06);
}
[data-testid="stSidebar"] .stMarkdown { font-size: 0.9rem; }

/* Tabs — cleaner */
.stTabs [data-baseweb="tab-list"] {
    gap: 6px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 10px 16px;
    font-weight: 500;
}

/* DataFrames — softer borders */
[data-testid="stDataFrame"] {
    border-radius: 12px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    overflow: hidden;
}

/* Divider — subtler */
hr {
    border-color: rgba(255, 255, 255, 0.07) !important;
    margin: 1rem 0 !important;
}

/* Code blocks (order specs) — monospace font + bordered */
code, pre {
    font-family: "JetBrains Mono", "SF Mono", Menlo, Monaco, monospace !important;
    font-size: 0.85rem !important;
}

/* Expanders */
.streamlit-expanderHeader {
    border-radius: 8px;
    font-weight: 500;
}

/* Section labels in sidebar */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    text-transform: uppercase;
    letter-spacing: 0.08em;
    font-size: 0.7rem !important;
    opacity: 0.6;
    margin-top: 8px;
}
</style>
"""


def inject_css() -> None:
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)


def is_mini() -> bool:
    """True when the app is opened in 'mini' mode (?mini=1) — e.g. inside a
    Telegram Mini App on a phone. Pages use this to switch to a stacked,
    touch-friendly layout. Desktop (no flag) is unaffected."""
    try:
        return str(st.query_params.get("mini", "")).lower() in ("1", "true", "yes")
    except Exception:
        return False


def init_session_state() -> None:
    if "results" not in st.session_state:
        st.session_state["results"] = {}   # symbol -> AnalysisResult
    if "errors" not in st.session_state:
        st.session_state["errors"] = {}


# ---------- formatting helpers ----------

def money(value: float, currency: str) -> str:
    sym = {"USD": "$", "HKD": "HK$"}.get(currency, "")
    return f"{sym}{value:,.2f}"


def signed_pct(value: float) -> str:
    sign = "+" if value >= 0 else ""
    return f"{sign}{value:.1f}%"


def level_color(label: str | None) -> str:
    return {
        "YELLOW_WATCH": "#f1c40f",
        "ORANGE_TRIM": "#e67e22",
        "RED_DEFENSIVE": "#e74c3c",
        "BLACK_EXIT": "#1a1a1a",
    }.get(label or "", "#27ae60")


def list_targets() -> list[tuple[str, str]]:
    """Held positions + configured watchlist + user-added tickers, de-duplicated."""
    portfolio = load_portfolio()
    universe = load_universe()
    user_tickers = load_user_tickers()
    seen: set[str] = set()
    targets: list[tuple[str, str]] = []
    for h in portfolio.holdings:
        if h.symbol not in seen:
            targets.append((h.symbol, h.market))
            seen.add(h.symbol)
    for market in universe.active_markets:
        for w in universe.watchlist.get(market, []):
            if w.symbol not in seen:
                targets.append((w.symbol, market))
                seen.add(w.symbol)
    for market in ("US", "HK"):
        for sym in user_tickers.get(market, []):
            if sym not in seen:
                targets.append((sym, market))
                seen.add(sym)
    return targets


def configured_symbols() -> set[str]:
    """Symbols that come from yaml configs (NOT removable from the dashboard)."""
    portfolio = load_portfolio()
    universe = load_universe()
    syms = {h.symbol for h in portfolio.holdings}
    for market_entries in universe.watchlist.values():
        for w in market_entries:
            syms.add(w.symbol)
    return syms


# ---------- pipeline progress runner ----------

_STEP_LABEL = {
    "data_fetch":       "Data fetch (prices + fundamentals + macro)",
    "competence_gate":  "Competence Gate",
    "fundamental":      "Fundamental Analyst",
    "valuation":        "Valuation",
    "financial_report": "Financial Report (last-2 reports + deep resolution)",
    "catalysts_data":   "Catalyst data (news + earnings + econ calendar)",
    "ir_agent":         "Information Retrieval",
    "forward_scenarios":"Forward Scenarios",
    "risk_analyzer":    "Risk Analyzer",
    "tactical_exit":    "Tactical Exit",
    "order_generator":  "Order Generator",
    "hedging":          "Hedging Agent",
    "contrarian":       "Contrarian Agent",
    "devil_advocate":   "Devil's Advocate",
    "audit_persist":    "Audit Trail Write",
}

_STATUS_EMOJI = {
    "pending":   "⋯",
    "started":   "⟳",
    "completed": "✓",
    "skipped":   "⊘",
    "failed":    "✗",
}


def _fmt_duration(seconds: float) -> str:
    if seconds < 1:
        return f"{seconds*1000:.0f}ms"
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}:{s:02d}"


def _render_progress(
    slot,
    sym: str,
    mkt: str,
    steps_state: dict[str, tuple[str, str]],
    step_durations: dict[str, float],
    step_started_at: dict[str, float],
) -> None:
    lines: list[str] = []
    now = time.time()
    for step in PIPELINE_STEPS:
        status, msg = steps_state.get(step, ("pending", ""))
        emoji = _STATUS_EMOJI.get(status, "·")
        label = _STEP_LABEL.get(step, step)

        duration_str = ""
        if status == "completed" and step in step_durations:
            duration_str = f" `{_fmt_duration(step_durations[step])}`"
        elif status == "started" and step in step_started_at:
            elapsed = now - step_started_at[step]
            duration_str = f" `{_fmt_duration(elapsed)}…`"

        line = f"{emoji} **{label}**{duration_str}"
        if msg:
            line += f"  · _{msg}_"
        lines.append(line)
    slot.markdown("\n\n".join(lines))


def run_selected(targets_to_run: list[tuple[str, str]]) -> None:
    """Run analysis on each (symbol, market) with a live progress panel per ticker."""
    if not targets_to_run:
        st.warning("No tickers selected.")
        return

    n = len(targets_to_run)

    past_durations: list[float] = st.session_state.get("ticker_durations", [])
    if past_durations:
        avg_duration = sum(past_durations[-10:]) / len(past_durations[-10:])
    else:
        avg_duration = 75.0  # rough default for one full pipeline pass
    estimated_total = avg_duration * n

    eta_header = st.empty()
    eta_header.info(
        f"**Running {n} ticker{'s' if n != 1 else ''}** · "
        f"Estimated total: **~{_fmt_duration(estimated_total)}** "
        f"(based on {'recent runs' if past_durations else 'default — first run'})"
    )

    progress_root = st.container()
    progress_root.markdown("### 🔬 Pipeline progress")

    overall_start = time.time()
    completed_durations: list[float] = []

    for i, (sym, mkt) in enumerate(targets_to_run):
        steps_state: dict[str, tuple[str, str]] = {step: ("pending", "") for step in PIPELINE_STEPS}
        step_started_at: dict[str, float] = {}
        step_durations: dict[str, float] = {}

        status_box = progress_root.status(
            f"⋯  {sym}  ({mkt}) — pending",
            state="running",
            expanded=True,
        )
        slot = status_box.empty()
        _render_progress(slot, sym, mkt, steps_state, step_durations, step_started_at)

        def make_cb(state_ref, started_at_ref, durations_ref):
            def cb(step: str, status: str, msg):
                state_ref[step] = (status, msg or "")
                if status == "started":
                    started_at_ref[step] = time.time()
                elif status in ("completed", "failed", "skipped") and step in started_at_ref:
                    durations_ref[step] = time.time() - started_at_ref[step]
                _render_progress(slot, sym, mkt, state_ref, durations_ref, started_at_ref)
            return cb

        ticker_start = time.time()
        status_box.update(label=f"⟳  {sym}  ({mkt}) — running", state="running")
        try:
            result = analyze(sym, mkt, on_progress=make_cb(steps_state, step_started_at, step_durations))
            ticker_dur = time.time() - ticker_start
            completed_durations.append(ticker_dur)
            st.session_state["results"][sym] = result
            st.session_state["errors"].pop(sym, None)
            verdict = ""
            if result.devil_advocate:
                verdict = f" · DA: {result.devil_advocate.overall_verdict}"
            status_box.update(
                label=f"✓  {sym}  ({mkt}) — complete in {_fmt_duration(ticker_dur)}{verdict}",
                state="complete",
                expanded=False,
            )
        except LLMConfigError as e:
            st.session_state["errors"][sym] = str(e)
            status_box.update(label=f"✗  {sym} — LLM config error: {e}", state="error", expanded=True)
        except Exception as e:
            logging.exception("Analysis failed for %s", sym)
            st.session_state["errors"][sym] = f"{type(e).__name__}: {e}"
            status_box.update(label=f"✗  {sym} — {type(e).__name__}: {e}", state="error", expanded=True)

        elapsed = time.time() - overall_start
        if completed_durations:
            avg_so_far = sum(completed_durations) / len(completed_durations)
            remaining = avg_so_far * (n - i - 1)
        else:
            remaining = avg_duration * (n - i - 1)
        eta_header.info(
            f"**Progress: {i+1}/{n} done** · "
            f"Elapsed: **{_fmt_duration(elapsed)}** · "
            f"Remaining: **~{_fmt_duration(remaining)}**"
        )

    if completed_durations:
        past_durations = past_durations + completed_durations
        st.session_state["ticker_durations"] = past_durations[-30:]
        total_elapsed = time.time() - overall_start
        eta_header.success(
            f"**Done.** {n} ticker{'s' if n != 1 else ''} analyzed in **{_fmt_duration(total_elapsed)}** "
            f"(avg {_fmt_duration(sum(completed_durations)/len(completed_durations))}/ticker)"
        )
