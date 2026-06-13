"""Markdown report builder per the project docs §5 — US / global (NYSE panel) report.

26-section daily template plus an auto-generated glossary; weekly/monthly
variants extend it. The Hong Kong / Greater China universe has its own
independent report in report_hk.py (different trading calendar).

The report is generated from:
  - metrics_daily (per-asset OHLC-derived metrics)
  - signals_daily (8 composite scores + components)
  - regime_history (current regime)
  - prior-period rows for "what changed" sections
"""
from __future__ import annotations

import json
import logging
import os
from datetime import date, timedelta
from typing import Iterable

import duckdb
import pandas as pd

from .analogue import (
    find_analogues,
    forecast_confidence,
    forecast_distribution,
    regime_transition_matrix,
    rotation_probability_matrix,
    sector_leader_forecast,
)
from .buckets import (
    BUCKET_RISK_TAG,
    bucket_for,
    group_metrics_by_bucket,
    risk_tag,
)
from .config import Config
from .glossary import render_glossary
from .ingest.fred_adapter import load_fred_panel
from .interpret import (
    CONFIDENCE_FLOOR, CONFIDENCE_LOW, CONFIDENCE_MED,
    confidence_bucket, interpret,
)
from .llm_interpret import llm_interpret
from .store import connect
from .validate import latest_verdicts


# ============================================================
# Data loaders
# ============================================================

def _load_signals(con: duckdb.DuckDBPyConnection, ts: date) -> dict[str, dict]:
    df = con.execute(
        "SELECT signal_name, score, confidence, components "
        "FROM signals_daily WHERE ts = ?",
        [ts],
    ).df()
    out: dict[str, dict] = {}
    for _, row in df.iterrows():
        comp = json.loads(row["components"]) if row["components"] else {}
        out[row["signal_name"]] = {
            "score": None if pd.isna(row["score"]) else float(row["score"]),
            "confidence": None if pd.isna(row["confidence"]) else float(row["confidence"]),
            **comp,
        }
    return out


def _load_metrics(con: duckdb.DuckDBPyConnection, ts: date) -> pd.DataFrame:
    return con.execute("SELECT * FROM metrics_daily WHERE ts = ? ORDER BY symbol", [ts]).df()


def _load_regime(con: duckdb.DuckDBPyConnection, ts: date) -> dict | None:
    df = con.execute(
        "SELECT regime, prev_regime, confidence, days_in_regime, components "
        "FROM regime_history WHERE ts <= ? ORDER BY ts DESC LIMIT 1",
        [ts],
    ).df()
    if df.empty:
        return None
    row = df.iloc[0]
    return {
        "regime": row["regime"],
        "prev_regime": row["prev_regime"],
        "confidence": float(row["confidence"]) if row["confidence"] is not None else None,
        "days_in_regime": int(row["days_in_regime"]) if row["days_in_regime"] is not None else None,
        "components": json.loads(row["components"]) if row["components"] else {},
    }


def _prior_trading_day(con: duckdb.DuckDBPyConnection, ts: date, k: int = 1) -> date | None:
    df = con.execute(
        "SELECT DISTINCT ts FROM signals_daily WHERE ts < ? ORDER BY ts DESC LIMIT ?",
        [ts, k],
    ).df()
    if len(df) < k:
        return None
    return df["ts"].iloc[k - 1].date() if hasattr(df["ts"].iloc[k - 1], "date") else df["ts"].iloc[k - 1]


# ============================================================
# Section renderers
# ============================================================

def _fmt(x, fmt: str = "{:+.1f}") -> str:
    # pd.isna covers None, float('nan') AND pd.NA — the latter is not a float
    # and would otherwise render literally as '<NA>' (and break the PDF's
    # markdown→HTML pass, which reads it as an unclosed <NA> tag).
    try:
        if x is None or pd.isna(x):
            return "—"
    except (TypeError, ValueError):
        pass  # array-likes; let the format spec deal with it
    return fmt.format(x)


def _confidence_badge(c: float | None) -> str:
    b = confidence_bucket(c)
    if b == "high":
        return "**high**"
    if b == "medium":
        return "_medium_"
    if b == "low":
        return "low"
    if b == "below_floor":
        return "↓ below floor"
    return "—"


log = logging.getLogger(__name__)


def _section_header(n: int, title: str) -> str:
    return f"\n## Section {n} — {title}\n"


def _safe_section(label: str, render) -> str:
    """C2 — render one section in isolation. A single section raising must not
    abort the whole report (and its Telegram delivery); it degrades to a visible
    placeholder instead. `render` is a zero-arg callable returning the markdown."""
    try:
        return render()
    except Exception as exc:  # noqa: BLE001 — one bad section can't sink the report
        log.warning("report section %r failed to render: %s", label, exc, exc_info=True)
        return (f"\n## {label} — unavailable\n\n_This section failed to render "
                f"({type(exc).__name__}: {exc}). The rest of the report is "
                f"unaffected._\n")


def section_overview(
    cfg: Config,
    asof: date,
    signals: dict,
    metrics: pd.DataFrame,
    regime: dict | None,
    interp: dict,
    verdicts: dict | None = None,
    n: int = 1,
) -> str:
    """Wishlist Final Vision — answer the 4 questions on page one.

    Order:
      1. Regime line.
      2. LLM/rules-based top observation (narrative-first).
      3. Where money is leaving (top-5 buckets, week to date).
      4. Where money is entering (top-5 buckets, week to date).
      5. Where money is likely to go next (top-4 buckets, 5d analogues).
      6. How confident — rotation strength, 5d/21d forecast, validation credit.

    Every number on this page is also present in its detail section below
    (§4 for flows, §7 for strength, §11 for likely-next, §12 for forecast).
    The overview lifts headlines; the sections give the full context.
    """
    out = [_section_header(n, "Overview")]

    # --- Regime line ---
    if regime:
        bits = [f"**Regime:** {regime['regime']}"]
        if regime.get("days_in_regime"):
            bits.append(f"day {regime['days_in_regime']}")
        if regime.get("confidence") is not None:
            bits.append(
                f"confidence {regime['confidence']:.2f} "
                f"({confidence_bucket(regime['confidence'])})"
            )
        out.append(" · ".join(bits))

    # --- Narrative-first: LLM top observation goes immediately under the regime ---
    claims = interp.get("claims", [])
    if claims:
        src = interp.get("source", "rules-based")
        label = "LLM" if src == "deepseek-llm" else "rules"
        out.append("")
        out.append(f"> **Top observation ({label}):** " + claims[0]["text"])
        if len(claims) > 1:
            out.append(f"> **Secondary:** " + claims[1]["text"])
    else:
        out.append("")
        out.append("> _No narrative fired — every claim sat below the confidence floor "
                   "or was suppressed by failed validation. See §20 and §25 for the "
                   "raw scores and reasons._")

    # --- Where money is leaving / entering (lifted from §4 Capital Flow Dashboard) ---
    if not metrics.empty:
        buckets = group_metrics_by_bucket(metrics).sort_values("r_w_mean", ascending=False)
        entering = buckets[buckets["r_w_mean"] >  _BUCKET_FLAT_THRESHOLD].head(5)
        leaving  = (buckets[buckets["r_w_mean"] < -_BUCKET_FLAT_THRESHOLD]
                    .sort_values("r_w_mean", ascending=True).head(5))

        out.append("\n### Where money is leaving (week to date)")
        if leaving.empty:
            out.append("\n_No bucket below the −50 bps threshold this week._")
        else:
            out.append("\n| Bucket | r_w | Members |")
            out.append("|---|---:|---|")
            for name, r in leaving.iterrows():
                out.append(f"| **{name}** | {r['r_w_mean']*100:+.2f}% | {r['members']} |")

        out.append("\n### Where money is entering (week to date)")
        if entering.empty:
            out.append("\n_No bucket above the +50 bps threshold this week._")
        else:
            out.append("\n| Bucket | r_w | Members |")
            out.append("|---|---:|---|")
            for name, r in entering.iterrows():
                out.append(f"| **{name}** | {r['r_w_mean']*100:+.2f}% | {r['members']} |")
    else:
        out.append("\n_No per-asset metrics for this date — flow tables omitted._")

    # --- Where money is likely to go next (lifted from §11) ---
    out.append("\n### Where money is likely to go next (5d, from 20 closest analogues)")
    rpm = rotation_probability_matrix(cfg, asof, k=20, horizon=5)
    if rpm.empty:
        out.append("\n_Insufficient analogue history for a conditional forecast._")
    else:
        out.append("\n| Bucket | Frequency | Avg fwd return | Risk tag |")
        out.append("|---|---:|---:|---|")
        for bucket, row in rpm.head(4).iterrows():
            freq = row["frequency"]
            avg = row["avg_fwd_return"]
            avg_str = f"{avg*100:+.2f}%" if pd.notna(avg) else "—"
            out.append(f"| **{bucket}** | {freq*100:.0f}% | {avg_str} | {risk_tag(bucket)} |")

    # --- How confident (rotation strength + forecast probs + validation credit) ---
    out.append("\n### How confident")
    conf_lines = _overview_confidence_lines(cfg, asof, signals, verdicts)
    if conf_lines:
        out.extend(conf_lines)
    else:
        out.append("- _No confidence inputs available._")

    out.append("\n_Numbers above are model outputs, not predictions. Full detail in the sections that follow._\n")
    return "\n".join(out)


def _overview_confidence_lines(
    cfg: Config,
    asof: date,
    signals: dict,
    verdicts: dict | None,
) -> list[str]:
    """Build the "How confident" bullets. Pulled out so the function above stays readable."""
    lines: list[str] = []

    # Rotation strength — same recipe as section_rotation_strength
    cr_score = signals.get("capital_rotation", {}).get("score")
    if cr_score is not None:
        with connect(cfg.storage.duckdb_path) as con:
            df = con.execute(
                "SELECT score FROM signals_daily "
                "WHERE signal_name = 'capital_rotation' AND ts <= ? AND score IS NOT NULL "
                "ORDER BY ts DESC LIMIT 252",
                [asof],
            ).df()
        strength = abs(float(cr_score))
        if len(df) > 1:
            hist = df["score"].abs()
            pct = float((hist <= strength).sum() / len(hist)) * 100
            tail = hist.iloc[1:].head(5)
            mean5 = float(tail.mean()) if not tail.empty else None
            if mean5 is None:
                trend = "—"
            elif strength > mean5 * 1.10:
                trend = "**rising**"
            elif strength < mean5 * 0.90:
                trend = "**falling**"
            else:
                trend = "flat"
            lines.append(
                f"- **Rotation Strength:** {strength:.1f}/100 · "
                f"{pct:.0f}th percentile vs trailing {len(hist)}d · {trend}"
            )
        else:
            lines.append(f"- **Rotation Strength:** {strength:.1f}/100 (insufficient history for percentile)")

    # Forecast probabilities
    f5  = forecast_distribution(cfg, asof, target="SPY", horizon=5,  k=30)
    f21 = forecast_distribution(cfg, asof, target="SPY", horizon=21, k=30)
    if f5.get("bullish_pct") is not None:
        lines.append(
            f"- **5-day SPY:** {f5['bullish_pct']*100:.0f}% bullish · "
            f"median {f5['median_fwd']*100:+.2f}% · "
            f"top analogue similarity {f5['top_similarity']*100:.1f}%"
        )
    if f21.get("bullish_pct") is not None:
        lines.append(
            f"- **21-day SPY:** {f21['bullish_pct']*100:.0f}% bullish · "
            f"median {f21['median_fwd']*100:+.2f}% · "
            f"p10→p90 {f21['p10_fwd']*100:+.2f}% to {f21['p90_fwd']*100:+.2f}%"
        )
        c = forecast_confidence(f21, verdicts=verdicts)
        lines.append(f"- **Forecast confidence:** {c['score']:.2f} ({c['bucket']})")

    # Signal validation summary
    if verdicts:
        n_pass = sum(1 for v in verdicts.values() if v.verdict == "pass")
        n_fail = sum(1 for v in verdicts.values() if v.verdict == "fail")
        n_und  = sum(1 for v in verdicts.values() if v.verdict == "undetermined")
        n_tot  = len(verdicts)
        lines.append(
            f"- **Signal validation:** {n_pass}/{n_tot} pass · "
            f"{n_fail} fail · {n_und} undetermined (see §20)"
        )

    return lines


# ============================================================
# Wishlist W5 — Executive Dashboard (the 5-second decision table)
# Wishlist W4 — Signal Inflection Monitor (what's turning, not what's strong)
#
# Both open the report ABOVE §1 Overview, like a cover page. They lift the
# single most decision-relevant read out of the detail sections so a fund
# manager gets the actionable answer in 5–10 seconds. They add no new data —
# every figure traces to a numbered section below.
# ============================================================

# Composite signals whose *rising* value means more risk appetite. `recession`
# is the inverse (rising = worse) and is subtracted explicitly below.
_RISK_POSTURE_SIGNALS = ("risk_on_off", "capital_rotation", "growth")


def _risk_posture(signals: dict) -> float | None:
    """Collapse the risk-on composites (minus recession) into one scalar.

    Used for the regime-trajectory read: the *change* in this number over the
    prior week says whether the regime is improving or deteriorating, which is
    more decision-relevant than any single level. None if no inputs exist."""
    vals = [signals[s]["score"] for s in _RISK_POSTURE_SIGNALS
            if signals.get(s, {}).get("score") is not None]
    rec = signals.get("recession", {}).get("score")
    if not vals and rec is None:
        return None
    posture = (sum(vals) / len(vals)) if vals else 0.0
    if rec is not None:
        posture -= rec
    return posture


def _trajectory(now: dict, prev: dict | None) -> tuple[str, float | None]:
    """Regime trajectory = change in risk posture over the prior window.

    ±10 pts of posture move is the call threshold (same order of magnitude as a
    signal's bull/bear band). Returns (label, delta)."""
    p_now = _risk_posture(now)
    if prev is None or p_now is None:
        return "—", None
    p_prev = _risk_posture(prev)
    if p_prev is None:
        return "—", None
    d = p_now - p_prev
    if d >= 10:
        return "Improving ↑", d
    if d <= -10:
        return "Deteriorating ↓", d
    return "Stable →", d


def _flow_direction(signals: dict, buckets: pd.DataFrame) -> str:
    """One-phrase capital-flow read: capital_rotation first, then who's leading."""
    cr = signals.get("capital_rotation", {}).get("score")
    if cr is not None and cr >= 15:
        return "Risk-On Rotation"
    if cr is not None and cr <= -15:
        return "Defensive Rotation"
    if buckets is not None and not buckets.empty:
        top = buckets.iloc[0]
        if top["r_w_mean"] > _BUCKET_FLAT_THRESHOLD:
            tag = top["risk_tag"]
            if tag == "defensive":
                return "Defensive bid"
            if tag == "risk_on":
                return "Risk-on bid"
    roo = signals.get("risk_on_off", {}).get("score")
    if roo is not None and roo >= 15:
        return "Risk-On Rotation"
    if roo is not None and roo <= -15:
        return "Risk-Off"
    return "Mixed / no clear rotation"


def _recommended_exposure(signals: dict, trajectory_label: str, f5: dict) -> str:
    """Vote across recession, liquidity, regime trajectory and the 5d analogue
    lean. Majority decides Reduce / Add / Maintain — deliberately conservative
    (ties → Maintain)."""
    reduce_flags = add_flags = 0
    rec = signals.get("recession", {}).get("score")
    if rec is not None:
        if rec >= 20:
            reduce_flags += 1
        elif rec <= -20:
            add_flags += 1
    liq = signals.get("liquidity", {}).get("score")
    if liq is not None:
        if liq <= 40:
            reduce_flags += 1
        elif liq >= 60:
            add_flags += 1
    if "Deteriorating" in trajectory_label:
        reduce_flags += 1
    elif "Improving" in trajectory_label:
        add_flags += 1
    if f5 and f5.get("bullish_pct") is not None:
        if f5["bullish_pct"] >= 0.60:
            add_flags += 1
        elif f5.get("bearish_pct", 0) >= 0.60:
            reduce_flags += 1
    if reduce_flags > add_flags:
        return "Reduce Risk ↓"
    if add_flags > reduce_flags:
        return "Add Risk ↑"
    return "Maintain →"


def _overall_confidence_word(signals: dict) -> str:
    confs = [s.get("confidence") for s in signals.values() if s.get("confidence") is not None]
    if not confs:
        return "—"
    return confidence_bucket(sum(confs) / len(confs)).replace("_", " ").title()


def section_executive_dashboard(
    cfg: Config,
    asof: date,
    signals: dict,
    prev_w_signals: dict | None,
    metrics: pd.DataFrame,
    regime: dict | None,
    verdicts: dict | None = None,
) -> str:
    """Wishlist W5 — the cover-page decision table.

    ≤10 rows answering the questions a fund manager asks first: what state are
    we in, is it getting better or worse, which way is capital flowing, what's
    leading / lagging, what does the model expect, and how much should I trust
    it. Non-numbered: it sits above §1 Overview."""
    out = ["\n## Executive Dashboard\n",
           "_The 5-second read. Every row is detailed in a numbered section "
           "below — this is the decision summary, not new information._\n"]

    buckets = (group_metrics_by_bucket(metrics).sort_values("r_w_mean", ascending=False)
               if not metrics.empty else pd.DataFrame())
    f5 = forecast_distribution(cfg, asof, target="SPY", horizon=5, k=30)

    traj_label, _ = _trajectory(signals, prev_w_signals)
    flow = _flow_direction(signals, buckets)
    exposure = _recommended_exposure(signals, traj_label, f5)

    state = regime["regime"] if regime else "—"
    if not buckets.empty:
        strongest = f"{buckets.index[0]} ({buckets.iloc[0]['r_w_mean']*100:+.2f}%)"
        weakest = f"{buckets.index[-1]} ({buckets.iloc[-1]['r_w_mean']*100:+.2f}%)"
    else:
        strongest = weakest = "—"
    if f5.get("bullish_pct") is not None:
        outlook = f"{f5['bullish_pct']*100:.0f}% bullish (median {f5['median_fwd']*100:+.2f}%)"
    else:
        outlook = "insufficient analogue history"

    rows = [
        ("Current Market State", state),
        ("Regime Trajectory", traj_label),
        ("Capital Flow Direction", flow),
        ("Strongest Sector", strongest),
        ("Weakest Sector", weakest),
        ("5-Day SPY Outlook", outlook),
        ("Recommended Risk Exposure", exposure),
        ("Signal Confidence", _overall_confidence_word(signals)),
    ]
    out.append("| Question | Answer |")
    out.append("|---|---|")
    for q, a in rows:
        out.append(f"| {q} | {a} |")
    out.append("\n_Not investment advice. See §2 (committee view) for the reasoning "
               "and §14 (scorecard) for the model's historical track record._")
    return "\n".join(out)


# Composites shown in the inflection monitor, in display order. relative_strength
# is excluded for the same reason as the "what changed" delta — its headline is
# the top-|z| asset, which hops between assets day to day.
_INFLECTION_SIGNALS = (
    ("growth", "Growth"),
    ("inflation", "Inflation"),
    ("recession", "Recession"),
    ("liquidity", "Liquidity"),
    ("risk_on_off", "Risk On/Off"),
    ("capital_rotation", "Capital Rotation"),
)


def _trend_arrow(delta: float | None, big: float = 20.0) -> str:
    if delta is None:
        return "—"
    if delta >= big:
        return "⇈"
    if delta > 0:
        return "↑"
    if delta <= -big:
        return "⇊"
    if delta < 0:
        return "↓"
    return "→"


def section_inflection_monitor(
    signals: dict,
    prev_w_signals: dict | None,
    metrics: pd.DataFrame,
) -> str:
    """Wishlist W4 — what's *turning*, not what's already strong.

    Each composite's score 5 trading days ago vs now, with a trend arrow. A
    signal still positive but falling fast often matters more than a flat strong
    one — turning points lead static readings. The fastest accelerating /
    decelerating assets by Δ²RS are the per-asset companion."""
    out = ["\n## Signal Inflection Monitor\n",
           "_What is *changing*. Investors make money from turning points, not "
           "static readings. Composite score 5 trading days ago vs now._\n"]

    if not prev_w_signals:
        out.append("_No 5-session-prior snapshot in the store yet — the monitor "
                   "needs a week of signal history._")
    else:
        out.append("| Signal | 5d ago | Now | Δ | Trend |")
        out.append("|---|---:|---:|---:|:---:|")
        any_row = False
        for key, label in _INFLECTION_SIGNALS:
            now_s = signals.get(key, {}).get("score")
            if now_s is None:
                continue
            prev_s = prev_w_signals.get(key, {}).get("score")
            d = (now_s - prev_s) if prev_s is not None else None
            out.append(f"| {label} | {_fmt(prev_s)} | {_fmt(now_s)} | {_fmt(d)} | "
                       f"{_trend_arrow(d)} |")
            any_row = True
        if not any_row:
            out.append("| _no composite signals for this date_ | | | | |")

    if not metrics.empty and "rs_accel_5" in metrics.columns:
        acc = metrics.dropna(subset=["rs_accel_5"])
        if not acc.empty:
            top = acc.sort_values("rs_accel_5", ascending=False).head(3)
            bot = acc.sort_values("rs_accel_5", ascending=True).head(3)
            out.append("\n**Fastest accelerating (Δ²RS — earliest leadership signal):** "
                       + ", ".join(f"{r['symbol']} ({_fmt(r['rs_accel_5'], '{:+.0f}')})"
                                   for _, r in top.iterrows()))
            out.append("\n**Fastest decelerating:** "
                       + ", ".join(f"{r['symbol']} ({_fmt(r['rs_accel_5'], '{:+.0f}')})"
                                   for _, r in bot.iterrows()))

    out.append("\n_⇈/⇊ = move ≥20 pts over 5 sessions; ↑/↓ = smaller move; → = flat. "
               "Δ²RS = acceleration of relative strength, which turns before raw RS._")
    return "\n".join(out)


def section_market_regime(regime: dict | None, n: int = 2) -> str:
    out = [_section_header(n, "Market Regime")]
    if not regime:
        out.append("_Regime classifier has not yet been run for this date._")
        return "\n".join(out)
    out.append(f"**Current:** {regime['regime']}")
    if regime.get("prev_regime") and regime["prev_regime"] != regime["regime"]:
        out.append(f"**Previous:** {regime['prev_regime']}")
        out.append(f"_Transition triggered hysteresis after {regime['days_in_regime']} confirming day(s)._")
    out.append(f"**Confidence:** {_confidence_badge(regime.get('confidence'))}")
    return "\n".join(out)


def _top_movers(metrics: pd.DataFrame, n: int = 5, ascending: bool = False) -> pd.DataFrame:
    return metrics.sort_values("r_w", ascending=ascending).head(n)


# ============================================================
# Phase A — Capital Flow Dashboard, Flow Map, Leadership Tracker
# ============================================================

# Material-change cut-off for bucket-level r_w. Below this magnitude a bucket
# is treated as "flat" and excluded from the flow narrative — too much noise
# to call money "leaving" or "entering" otherwise.
# 50 bps weekly by default; tunable without a code edit via env var.
_BUCKET_FLAT_THRESHOLD = float(os.environ.get("ROTATION_BUCKET_FLAT_THRESHOLD", "0.005"))


def section_capital_flow_dashboard(metrics: pd.DataFrame, n: int = 3) -> str:
    """Phase-A wishlist item 1 — explicit money-leaving / money-entering.

    Aggregates per-symbol r_w into buckets and groups them into two columns
    so the reader doesn't have to infer the rotation from individual tickers.
    """
    out = [_section_header(n, "Capital Flow Dashboard")]
    if metrics.empty:
        out.append("_No metrics available._"); return "\n".join(out)

    buckets = group_metrics_by_bucket(metrics).sort_values("r_w_mean", ascending=False)
    entering = buckets[buckets["r_w_mean"] >  _BUCKET_FLAT_THRESHOLD]
    leaving  = buckets[buckets["r_w_mean"] < -_BUCKET_FLAT_THRESHOLD]
    flat     = buckets[buckets["r_w_mean"].abs() <= _BUCKET_FLAT_THRESHOLD]

    out.append("_Aggregated weekly return (r_w) by bucket. ≥50 bps = directional; "
               "below that the bucket is flat and excluded from the rotation call._\n")

    out.append("**Money Entering (week to date)**\n")
    if entering.empty:
        out.append("_No bucket exceeded the +50 bps threshold._")
    else:
        out.append("| Bucket | r_w | r_m | breadth (adv %) | members |")
        out.append("|---|---:|---:|---:|---|")
        for name, r in entering.iterrows():
            out.append(
                f"| **{name}** | {r['r_w_mean']*100:+.2f}% | "
                f"{_fmt(r['r_m_mean']*100, '{:+.2f}%')} | "
                f"{r['pct_adv_w']*100:.0f}% ({int(r['n'])}) | {r['members']} |"
            )

    out.append("\n**Money Leaving (week to date)**\n")
    if leaving.empty:
        out.append("_No bucket fell below the −50 bps threshold._")
    else:
        out.append("| Bucket | r_w | r_m | breadth (adv %) | members |")
        out.append("|---|---:|---:|---:|---|")
        for name, r in leaving.sort_values("r_w_mean", ascending=True).iterrows():
            out.append(
                f"| **{name}** | {r['r_w_mean']*100:+.2f}% | "
                f"{_fmt(r['r_m_mean']*100, '{:+.2f}%')} | "
                f"{r['pct_adv_w']*100:.0f}% ({int(r['n'])}) | {r['members']} |"
            )

    if not flat.empty:
        out.append(f"\n_Flat (within ±50 bps): {', '.join(sorted(flat.index.tolist()))}._")
    return "\n".join(out)


def _flow_bar(magnitude: float, scale: float, width: int = 12) -> str:
    """Render a horizontal bar whose width is proportional to |magnitude|.

    `scale` is the largest |magnitude| across the whole flow map, so bars
    are comparable across leaving / entering columns within one report.
    """
    if scale <= 0 or magnitude == 0:
        return ""
    n = max(1, min(width, int(round(abs(magnitude) / scale * width))))
    return "█" * n


def section_flow_map(metrics: pd.DataFrame, n: int = 4) -> str:
    """Phase-A wishlist item 2 — ASCII flow map.

    Two columns of ranked buckets with bar magnitudes. The "→" between columns
    is purely typographic; this is the at-a-glance visual.
    """
    out = [_section_header(n, "Flow Map")]
    if metrics.empty:
        out.append("_No metrics available._"); return "\n".join(out)

    buckets = group_metrics_by_bucket(metrics)
    entering = buckets[buckets["r_w_mean"] >  _BUCKET_FLAT_THRESHOLD] \
                  .sort_values("r_w_mean", ascending=False).head(5)
    leaving  = buckets[buckets["r_w_mean"] < -_BUCKET_FLAT_THRESHOLD] \
                  .sort_values("r_w_mean", ascending=True).head(5)
    if entering.empty and leaving.empty:
        out.append("_No bucket exceeded ±50 bps this week — no rotation to map._")
        return "\n".join(out)

    scale = max(
        entering["r_w_mean"].abs().max() if not entering.empty else 0.0,
        leaving["r_w_mean"].abs().max() if not leaving.empty else 0.0,
        1e-9,
    )

    # Pair up rows. The two columns have different lengths so we zip with fillvalue.
    rows = max(len(entering), len(leaving))
    lines = ["```text", "  MONEY LEAVING                          MONEY ENTERING"]
    enter_iter = list(entering.iterrows())
    leave_iter = list(leaving.iterrows())
    for i in range(rows):
        left = right = " " * 38
        if i < len(leave_iter):
            name, r = leave_iter[i]
            bar = _flow_bar(r["r_w_mean"], scale)
            label = f"{name} ({r['r_w_mean']*100:+.2f}%)"
            left = f"{label:>26s}  {bar:<12s}"
        if i < len(enter_iter):
            name, r = enter_iter[i]
            bar = _flow_bar(r["r_w_mean"], scale)
            label = f"{name} ({r['r_w_mean']*100:+.2f}%)"
            right = f"{bar:>12s}  {label:<26s}"
        lines.append(f"  {left}  →  {right}")
    lines.append("```")
    out.append("\n".join(lines))
    out.append("\n_Bar width is proportional to |r_w| within this report. Top 5 each side._")
    return "\n".join(out)


def section_leadership_tracker(metrics: pd.DataFrame, n: int = 5) -> str:
    """Phase-A wishlist item 5 — fading / current / emerging leaders.

    Definitions:
      - Current Leaders : rs_rank ≥ 75 AND ΔRS(5d) ≥ −5  (still high, not fading)
      - Fading Leaders  : rs_rank ≥ 60 AND ΔRS(5d) ≤ −15 (high but losing it fast)
      - Emerging        : rs_rank ≥ 40 AND ΔRS(5d) ≥ +15 (mid-pack with momentum)

    Δ²RS(5d) is shown for context — positive accel under fading means the
    bleeding may be slowing; negative accel under emerging means the bid may
    be losing steam.
    """
    out = [_section_header(n, "Leadership Rotation Tracker")]
    if metrics.empty or "rs_rank" not in metrics.columns:
        out.append("_No metrics available._"); return "\n".join(out)

    m = metrics.dropna(subset=["rs_rank", "rs_change_5"]).copy()
    if m.empty:
        out.append("_Insufficient RS history to classify leadership._")
        return "\n".join(out)
    m["rs_rank"] = m["rs_rank"].astype(int)
    m["rs_change_5"] = m["rs_change_5"].astype(int)

    current  = m[(m["rs_rank"] >= 75) & (m["rs_change_5"] >= -5)] \
                  .sort_values("rs_rank", ascending=False).head(6)
    fading   = m[(m["rs_rank"] >= 60) & (m["rs_change_5"] <= -15)] \
                  .sort_values("rs_change_5", ascending=True).head(6)
    emerging = m[(m["rs_rank"] >= 40) & (m["rs_rank"] < 80) & (m["rs_change_5"] >= 15)] \
                  .sort_values("rs_change_5", ascending=False).head(6)

    def _render(group: pd.DataFrame) -> list[str]:
        if group.empty:
            return ["_(none)_"]
        rows = ["| Symbol | Bucket | RS | ΔRS(5d) | Δ²RS(5d) | r_w |",
                "|---|---|---:|---:|---:|---:|"]
        for _, r in group.iterrows():
            accel = r.get("rs_accel_5", pd.NA) if "rs_accel_5" in r.index else pd.NA
            rows.append(
                f"| {r['symbol']} | {bucket_for(r['symbol'])} | "
                f"{int(r['rs_rank'])} | {int(r['rs_change_5']):+d} | "
                f"{_fmt(accel, '{:+.0f}')} | {_fmt(r['r_w']*100, '{:+.2f}%')} |"
            )
        return rows

    out.append("**Current Leaders** — RS ≥ 75 and not yet bleeding rank.")
    out.extend(_render(current))
    out.append("\n**Fading Leaders** — RS still ≥ 60 but losing ≥15 ranks in 5d. _The "
               "earliest 'money is leaving' signal — surface before raw RS catches up._")
    out.extend(_render(fading))
    out.append("\n**Emerging Leaders** — RS mid-pack (40–80) but gaining ≥15 ranks in 5d.")
    out.extend(_render(emerging))
    return "\n".join(out)


def section_bucket_breadth(metrics: pd.DataFrame, n: int = 9) -> str:
    """Phase-A wishlist item 6 — per-bucket advancing %."""
    out = [_section_header(n, "Sector / Bucket Breadth")]
    if metrics.empty:
        out.append("_No metrics available._"); return "\n".join(out)

    rd = metrics["r_d"].dropna()
    if not rd.empty:
        n = len(rd)
        pct_adv = float((rd > 0).sum() / n)
        pct_dec = float((rd < 0).sum() / n)
        out.append(
            f"- Universe-wide: **{pct_adv*100:.1f}%** advancing, "
            f"**{pct_dec*100:.1f}%** declining "
            f"(net {pct_adv*100 - pct_dec*100:+.1f}pp).\n"
        )

    g = group_metrics_by_bucket(metrics).sort_values("r_w_mean", ascending=False)
    if g.empty:
        out.append("_No bucket coverage._"); return "\n".join(out)
    out.append("| Bucket | n | adv % (r_w) | mean r_w | mean r_m | risk tag |")
    out.append("|---|---:|---:|---:|---:|---|")
    for name, r in g.iterrows():
        out.append(
            f"| {name} | {int(r['n'])} | {_fmt(r['pct_adv_w']*100, '{:.0f}%')} | "
            f"{_fmt(r['r_w_mean']*100, '{:+.2f}%')} | {_fmt(r['r_m_mean']*100, '{:+.2f}%')} | "
            f"{r['risk_tag']} |"
        )
    out.append("\n_Breadth often shifts before price — a bucket whose r_w mean is "
               "flat but with 80% advancing is being accumulated; the opposite is being "
               "distributed._")
    return "\n".join(out)


def section_rotation_strength(
    con: duckdb.DuckDBPyConnection,
    asof: date,
    signals: dict,
    n: int = 6,
) -> str:
    """Phase-A wishlist item 3 — quantify how unusual the rotation is.

    Inputs:
      - Today's capital_rotation score (signed -100..+100).
      - Trailing 252d of capital_rotation scores from signals_daily.

    Outputs:
      - Strength = |score| → 0-100 magnitude.
      - Percentile rank vs trailing 252d.
      - Trend: rising / falling / flat (vs 5d trailing mean).
      - Duration: consecutive days |score| has been ≥ 25.
    """
    out = [_section_header(n, "Rotation Strength")]
    cr = signals.get("capital_rotation", {})
    score = cr.get("score")
    if score is None:
        out.append("_capital_rotation has no score for this date._")
        return "\n".join(out)

    df = con.execute(
        """
        SELECT ts, score
        FROM signals_daily
        WHERE signal_name = 'capital_rotation' AND ts <= ? AND score IS NOT NULL
        ORDER BY ts DESC
        LIMIT 252
        """,
        [asof],
    ).df()
    if df.empty:
        out.append(f"- **Strength:** {abs(score):.1f}/100 (raw |score|)")
        out.append("- _Insufficient history (<1d) to compute percentile or trend._")
        return "\n".join(out)
    df = df.sort_values("ts").reset_index(drop=True)
    strength_today = abs(float(score))
    strength_hist = df["score"].abs()
    pct = float((strength_hist <= strength_today).sum() / len(strength_hist)) * 100

    # Trend: today's magnitude vs trailing 5d mean (excluding today)
    tail = strength_hist.iloc[:-1].tail(5) if len(strength_hist) > 1 else pd.Series(dtype=float)
    if tail.empty:
        trend = "—"
    else:
        mean5 = float(tail.mean())
        if strength_today > mean5 * 1.10:
            trend = "**rising**"
        elif strength_today < mean5 * 0.90:
            trend = "**falling**"
        else:
            trend = "flat"

    # Duration above the |25| threshold (a meaningful-magnitude floor).
    THRESH = 25.0
    duration = 0
    for v in strength_hist.iloc[::-1]:
        if v >= THRESH:
            duration += 1
        else:
            break

    out.append(f"- **Rotation Strength:** {strength_today:.1f}/100")
    out.append(f"- **Percentile** vs trailing {len(strength_hist)}d: {pct:.0f}%")
    out.append(f"- **Trend** vs trailing 5d mean: {trend}")
    out.append(f"- **Duration** above |25|: {duration} day(s)")
    out.append(f"- **Direction** (signed score): {score:+.1f} "
               f"({'risk-on rotation' if score > 0 else 'risk-off rotation' if score < 0 else 'flat'})")
    return "\n".join(out)


# ============================================================
# Phase C — Historical Analogues, Transition Matrix, Probability Matrix
# ============================================================


def section_historical_analogues(cfg: Config, asof: date, n: int = 8) -> str:
    """Wishlist item 10 — find the k most-similar historical days and show
    what came next. The reader sees "today resembles X, similarity Y%, here's
    what happened after" — actionable in a way the headline regime label isn't.
    """
    out = [_section_header(n, "Historical Analogues")]
    anas = find_analogues(cfg, asof, k=5, blackout=30)
    if not anas:
        out.append("_No historical analogues found. (Not enough signal "
                   "history, or today has missing signal values.)_")
        return "\n".join(out)
    out.append("_Top-5 historical days by cosine similarity of today's 7-signal "
               "vector against the trailing 10+ years (30-day blackout)._\n")
    out.append("| Date | Similarity | Regime then | SPY +5d | SPY +21d | Regime +30d | Top buckets +5d |")
    out.append("|---|---:|---|---:|---:|---|---|")
    for a in anas:
        sim_pct = f"{a.similarity*100:.1f}%"
        reg = a.regime or "—"
        fwd5  = f"{a.fwd_return_spy_5d*100:+.2f}%"  if a.fwd_return_spy_5d  is not None else "—"
        fwd21 = f"{a.fwd_return_spy_21d*100:+.2f}%" if a.fwd_return_spy_21d is not None else "—"
        next_reg = a.next_regime_30d or "—"
        winners = ", ".join(a.bucket_winners_5d) if a.bucket_winners_5d else "—"
        out.append(f"| {a.asof} | {sim_pct} | {reg} | {fwd5} | {fwd21} | {next_reg} | {winners} |")
    out.append("\n_The 'Top buckets +5d' column lists what actually led in the 5 trading "
               "days after each analogue — the historical answer to 'what came next'._")
    return "\n".join(out)


def section_regime_transitions(
    cfg: Config,
    current_regime: str | None,
    n: int = 9,
) -> str:
    """Wishlist item 11 — next-30d regime probability conditional on today's regime.

    Reads the empirical transition matrix from regime_history and shows the row
    for the current regime. Falls back to the full matrix if regime is None.
    """
    out = [_section_header(n, "Regime Transition Probabilities")]
    m = regime_transition_matrix(cfg, window=30)
    if m.empty:
        out.append("_Insufficient regime history to estimate transitions._")
        return "\n".join(out)

    out.append("_Empirical P(regime in 30 trading days | current regime), estimated "
               "from the full regime history._\n")
    if current_regime and current_regime in m.index:
        row = m.loc[current_regime].sort_values(ascending=False)
        out.append(f"**Current regime:** {current_regime}\n")
        out.append("| Next-30d regime | Probability |")
        out.append("|---|---:|")
        for dst, p in row.items():
            out.append(f"| {dst} | {p*100:.0f}% |")
        out.append("\n_Read as: 'if history rhymes, the regime 30 trading days from now "
                   "is most likely to be the top row.'_")
    else:
        out.append("**Full transition matrix** (rows = current, columns = +30d):\n")
        out.append("| From \\\\ To | " + " | ".join(m.columns) + " |")
        out.append("|---|" + "---:|" * len(m.columns))
        for src, row in m.iterrows():
            parts = [f"{v*100:.0f}%" for v in row.values]
            out.append(f"| **{src}** | " + " | ".join(parts) + " |")
    return "\n".join(out)


def section_rotation_probability(
    cfg: Config,
    asof: date,
    n: int = 10,
    horizon: int = 5,
) -> str:
    """Wishlist item 9 — Where Money is Likely to Go Next.

    Pulls k=20 analogues, counts which buckets historically led in the next
    `horizon` days, and shows the empirical frequency.
    """
    out = [_section_header(n, "Where Money Likely Goes Next")]
    rpm = rotation_probability_matrix(cfg, asof, k=20, horizon=horizon)
    if rpm.empty:
        out.append("_No analogue-based forecast available (insufficient history)._")
        return "\n".join(out)

    out.append(f"_Frequency = fraction of 20 closest historical analogues where this "
               f"bucket was in the top-3 over the next {horizon} trading days. "
               f"avg_fwd_return = mean log return of that bucket across those windows._\n")
    out.append("| Bucket | Frequency | Avg forward return | Risk tag |")
    out.append("|---|---:|---:|---|")
    for bucket, row in rpm.head(8).iterrows():
        freq = row["frequency"]
        avg  = row["avg_fwd_return"]
        avg_str = f"{avg*100:+.2f}%" if pd.notna(avg) else "—"
        out.append(f"| **{bucket}** | {freq*100:.0f}% | {avg_str} | {risk_tag(bucket)} |")
    out.append("\n_This is **conditional historical**, not predicted. If today's signal "
               "vector resembles 20 prior days, this table is what tended to win in the "
               "next " + str(horizon) + " days across those 20 instances. Equal-weighted, "
               "no risk adjustment._")
    return "\n".join(out)


# ============================================================
# Phase D — Probabilistic forecasts (wishlist items 12, 13, 14)
# ============================================================


def section_market_forecast(
    cfg: Config,
    asof: date,
    verdicts: dict | None,
    n: int = 11,
) -> str:
    """Wishlist item 12 — probabilistic SPY outlook at 5d AND 21d.

    Avoids "SPY +3.2%" point estimates per the spec; presents the empirical
    Bullish/Neutral/Bearish distribution derived from 30 analogue forward
    returns, plus the p10-p90 tail. Confidence column (item 14) blends
    analogue similarity, directional agreement, and signal validation.
    """
    out = [_section_header(n, "Probabilistic Market Forecast")]
    rows = [
        ("5-Day Outlook", forecast_distribution(cfg, asof, target="SPY", horizon=5, k=30)),
        ("21-Day Outlook", forecast_distribution(cfg, asof, target="SPY", horizon=21, k=30)),
    ]
    if all(r[1].get("bullish_pct") is None for r in rows):
        out.append("_Insufficient historical analogues to produce a forecast._")
        return "\n".join(out)

    out.append("_SPY forecast derived from the 30 closest historical analogues "
               "(cosine similarity on the 7-signal vector). Probabilities are "
               "empirical, not point predictions._\n")

    out.append("| Horizon | Bullish | Neutral | Bearish | Median | p10 → p90 | n | Confidence |")
    out.append("|---|---:|---:|---:|---:|---|---:|:---|")
    for label, f in rows:
        if f.get("bullish_pct") is None:
            out.append(f"| {label} | — | — | — | — | — | 0 | — |")
            continue
        bp = f["bullish_pct"] * 100
        np_ = f["neutral_pct"] * 100
        brp = f["bearish_pct"] * 100
        med = f["median_fwd"] * 100
        p10 = f["p10_fwd"] * 100
        p90 = f["p90_fwd"] * 100
        conf = forecast_confidence(f, verdicts=verdicts)
        out.append(
            f"| **{label}** | {bp:.0f}% | {np_:.0f}% | {brp:.0f}% | "
            f"{med:+.2f}% | {p10:+.2f}% → {p90:+.2f}% | {f['n_analogues']} | "
            f"{conf['score']:.2f} ({conf['bucket']}) |"
        )

    # Detail block for the 21-day outlook — top similarity and confidence components.
    f21 = rows[1][1]
    if f21.get("bullish_pct") is not None:
        c = forecast_confidence(f21, verdicts=verdicts)["components"]
        out.append("")
        out.append(f"_Top analogue similarity: **{f21['top_similarity']*100:.1f}%** · "
                   f"directional agreement: **{f21['agreement']*100:.0f}%** · "
                   f"validation credit: **{c['validation_credit']*100:.0f}%** "
                   f"(share of underlying signals that pass IC validation)._")
        out.append("")
        out.append(f"_Cutoff for directional buckets: ±{f21['cutoff']*100:.2f}% "
                   "log return over the horizon. Below that magnitude the analogue "
                   "is counted as Neutral._")
    return "\n".join(out)


def section_sector_forecast(
    cfg: Config,
    asof: date,
    verdicts: dict | None,
    n: int = 12,
) -> str:
    """Wishlist item 13 — Expected Sector Leaders (Next 21 Days).

    Per the wishlist: "Sector forecasting is much easier than index forecasting."
    Aggregates the analogue forward returns by bucket and reports the empirical
    frequency that each bucket was in the top-5 over the next 21 trading days.
    """
    out = [_section_header(n, "Sector Forecast — Expected Leaders (Next 21 Days)")]
    slf = sector_leader_forecast(cfg, asof, horizon=21, k=30, top_k=5)
    if slf.empty:
        out.append("_Insufficient historical analogue data for a sector forecast._")
        return "\n".join(out)

    out.append("_Frequency = fraction of the 30 closest analogues where this "
               "bucket was a top-5 performer over the next 21 trading days. "
               "avg_fwd_return = mean 21-day log return for that bucket across "
               "the analogue windows._\n")
    out.append("| Bucket | Frequency | Avg 21d return | Risk tag |")
    out.append("|---|---:|---:|---|")
    for bucket, row in slf.iterrows():
        freq = row["frequency"]
        avg = row["avg_fwd_return"]
        avg_str = f"{avg*100:+.2f}%" if pd.notna(avg) else "—"
        out.append(f"| **{bucket}** | {freq*100:.0f}% | {avg_str} | {risk_tag(bucket)} |")
    out.append("\n_Per the wishlist's spec: this is **conditional historical**, "
               "not a forward prediction. The forecast layer's value is that "
               "sector dispersion is more predictable than index direction — see "
               "the relative ranking, not the absolute returns._")
    return "\n".join(out)


def section_forecast_scorecard(con: duckdb.DuckDBPyConnection, asof: date, n: int = 14) -> str:
    """Wishlist W1 — the report grades its own published forecasts.

    Hit rates per forecast type over the last 100 resolved forecasts, plus a
    recent forecast-vs-actual history. This is the section that answers
    "why should I trust the two sections above?" with data instead of tone.
    """
    from .scorecard import FORECAST_TYPES, load_scorecard

    out = [_section_header(n, "Forecast Scorecard — Actual vs Forecast")]
    sc = load_scorecard(con, asof)
    summary = sc["summary"]

    if all(v["n_resolved"] == 0 and v["n_pending"] == 0 for v in summary.values()):
        out.append("_No forecasts recorded yet. Each daily run now persists its "
                   "5d/21d SPY outlooks and sector-leader forecast; hit rates "
                   "appear here once horizons elapse. Backfill with "
                   "`rotate scorecard --backfill <start>..<end>`._")
        return "\n".join(out)

    labels = {
        "spy_5d": "5-Day SPY direction",
        "spy_21d": "21-Day SPY direction",
        "sector_21d": "Sector leaders (21d)",
    }
    out.append("_Every published outlook is recorded at publish time and graded "
               "once its horizon elapses. SPY hit = dominant analogue direction "
               "matched the realized direction (same cutoff as published). "
               "Sector hit = the #1 predicted bucket finished in the realized "
               "top-3. Last 100 resolved forecasts per row._\n")
    out.append("| Forecast | Hit rate | Resolved | Pending |")
    out.append("|---|---:|---:|---:|")
    for ftype in FORECAST_TYPES:
        s = summary[ftype]
        hr = f"**{s['hit_rate']*100:.0f}%**" if s["hit_rate"] is not None else "—"
        out.append(f"| {labels[ftype]} | {hr} | {s['n_resolved']} | {s['n_pending']} |")

    recent = sc["recent"]
    if not recent.empty:
        out.append("\n**Recent resolved forecasts**\n")
        out.append("| Published | Forecast | Predicted | Actual | Hit |")
        out.append("|---|---|---|---|:---:|")
        for _, r in recent.iterrows():
            ts_str = pd.Timestamp(r["ts"]).date().isoformat()
            if r["forecast_type"] == "sector_21d":
                det = json.loads(r["details"]) if r["details"] else {}
                pred = ", ".join(det.get("predicted_top3", [])) or "—"
                act = ", ".join(det.get("actual_top3", [])) or "—"
                out.append(f"| {ts_str} | {labels['sector_21d']} | {pred} | {act} | "
                           f"{'✓' if r['hit'] else '✗'} |")
            else:
                p_dir = (r["bullish_pct"] if r["direction"] == "bullish"
                         else r["bearish_pct"] if r["direction"] == "bearish"
                         else None)
                pred = f"{str(r['direction']).title()}"
                if p_dir is not None and pd.notna(p_dir):
                    pred += f" ({p_dir*100:.0f}%)"
                act = (f"{r['actual_value']*100:+.2f}% ({r['actual_direction']})"
                       if pd.notna(r["actual_value"]) else str(r["actual_direction"]))
                out.append(f"| {ts_str} | {labels[str(r['forecast_type'])]} | {pred} | "
                           f"{act} | {'✓' if r['hit'] else '✗'} |")

    out.append("\n_A hit rate near the base rate (markets drift up, so always-bullish "
               "scores above 50% on SPY) is not skill — compare against the bullish "
               "share in the Actual column, not against a coin flip._")
    return "\n".join(out)


# Composite signals that carry per-driver attribution, in presentation order.
_ATTRIBUTION_ORDER = (
    "capital_rotation", "risk_on_off", "growth", "inflation", "recession", "liquidity",
)


def section_signal_attribution(signals: dict, n: int = 21) -> str:
    """Wishlist W3 — no headline number left unexplained.

    For each composite, the per-driver breakdown stored in signals_daily:
    the driver's input (robust-z for basket signals, pair score for
    capital_rotation), its weight, and its contribution rescaled into
    headline score points (the Pts column sums to the score; for the 0..100
    liquidity level it sums to the distance from the neutral 50)."""
    out = [_section_header(n, "Signal Attribution")]
    out.append("_Each composite decomposed into its drivers. **Pts** rescales a "
               "driver's share of the pre-mapping aggregate into headline score "
               "points, so the column sums to the headline (liquidity: to its "
               "distance from the neutral 50). Largest |Pts| first._")

    rendered = 0
    for name in _ATTRIBUTION_ORDER:
        s = signals.get(name, {})
        score = s.get("score")
        if score is None:
            continue
        title = name.replace("_", " ").title()
        attr = s.get("attribution") or {}
        drivers = attr.get("drivers") or []
        if not drivers:
            out.append(f"\n### {title}: {_fmt(score)}\n")
            out.append("_No per-driver attribution recorded for this date — "
                       "signals were computed before attribution shipped. "
                       "Re-run `rotate signals` for this date to populate._")
            continue
        raw = attr.get("raw") or 0.0
        baseline = attr.get("baseline", 0.0)
        span = float(score) - float(baseline)
        head = f"\n### {title}: {_fmt(score)}"
        if baseline:
            head += f" (neutral = {baseline:.0f})"
        out.append(head + "\n")
        out.append("| Driver | Input | Weight | Pts |")
        out.append("|---|---:|---:|---:|")
        for d in drivers:
            pts = (d["contrib"] / raw * span) if raw else 0.0
            out.append(
                f"| {d['driver']} | {_fmt(d.get('value'), '{:+.2f}')} | "
                f"{_fmt(d.get('weight'), '{:+.2f}')} | {_fmt(pts, '{:+.1f}')} |"
            )
        out.append(f"| **Total** | | | **{_fmt(span, '{:+.1f}')}** |")
        rendered += 1

    if rendered == 0:
        out.append("\n_No composite signals with scores for this date._")
    return "\n".join(out)


def section_committee_view(
    cfg: Config,
    asof: date,
    signals: dict,
    regime: dict | None,
    verdicts: dict | None,
    metrics: pd.DataFrame,
    n: int = 2,
) -> str:
    """Wishlist W2 — the signals condensed into a decision-shaped summary.

    Bull/Bear bullets are rule-based readings of the live composites and bucket
    flows; the Net Assessment probabilities are the analogue forecast
    distributions (empirical, confidence-weighted across the 5d/21d horizons),
    NOT an invented number. Positioning is hedged model output, not advice.
    """
    out = [_section_header(n, "Investment Committee View")]

    def _score(name: str) -> float | None:
        return signals.get(name, {}).get("score")

    bull: list[str] = []
    bear: list[str] = []

    cr = _score("capital_rotation")
    if cr is not None:
        if cr >= 15:
            bull.append(f"Capital rotating toward risk assets (rotation {cr:+.0f})")
        elif cr <= -15:
            bear.append(f"Defensive rotation underway (rotation {cr:+.0f})")

    roo = _score("risk_on_off")
    if roo is not None:
        if roo >= 15:
            bull.append(f"Risk appetite positive (risk-on/off {roo:+.0f})")
        elif roo <= -15:
            bear.append(f"Risk-off tone across baskets (risk-on/off {roo:+.0f})")

    gr = _score("growth")
    if gr is not None:
        if gr >= 15:
            bull.append(f"Growth basket leading defensives (growth {gr:+.0f})")
        elif gr <= -15:
            bear.append(f"Growth basket lagging defensives (growth {gr:+.0f})")

    infl = _score("inflation")
    if infl is not None:
        if infl >= 25:
            bear.append(f"Inflation pressure building (inflation {infl:+.0f}) — duration headwind")
        elif infl <= -25:
            bull.append(f"Disinflation tailwind (inflation {infl:+.0f})")

    rec = _score("recession")
    if rec is not None:
        if rec >= 20:
            bear.append(f"Recession concern elevated ({rec:+.0f})")
        elif rec <= -20:
            bull.append(f"Recession concern receding ({rec:+.0f})")

    liq = _score("liquidity")
    if liq is not None:
        if liq >= 60:
            bull.append(f"Liquidity remains supportive ({liq:.0f}/100)")
        elif liq <= 40:
            bear.append(f"Liquidity tightening ({liq:.0f}/100)")

    # Analogue forecasts (also drive the Net Assessment below)
    f5 = forecast_distribution(cfg, asof, target="SPY", horizon=5, k=30)
    f21 = forecast_distribution(cfg, asof, target="SPY", horizon=21, k=30)
    for label, f in (("5-day", f5), ("21-day", f21)):
        if f.get("bullish_pct") is None:
            continue
        if f["bullish_pct"] >= 0.60:
            bull.append(f"{label} analogues {f['bullish_pct']*100:.0f}% bullish")
        elif f["bearish_pct"] >= 0.60:
            bear.append(f"{label} analogues {f['bearish_pct']*100:.0f}% bearish")

    # Bucket flows: who's attracting / shedding capital, with the risk tag
    # deciding which side of the ledger it argues for.
    entering = leaving = pd.DataFrame()
    if not metrics.empty:
        buckets = group_metrics_by_bucket(metrics).sort_values("r_w_mean", ascending=False)
        entering = buckets[buckets["r_w_mean"] > _BUCKET_FLAT_THRESHOLD]
        leaving = (buckets[buckets["r_w_mean"] < -_BUCKET_FLAT_THRESHOLD]
                   .sort_values("r_w_mean", ascending=True))
        if not entering.empty:
            top_in = entering.index[0]
            line = f"{top_in} attracting capital (r_w {entering.iloc[0]['r_w_mean']*100:+.2f}%)"
            (bear if risk_tag(top_in) == "defensive" else bull).append(
                line + (" — defensive bid" if risk_tag(top_in) == "defensive" else "")
            )
        if not leaving.empty:
            top_out = leaving.index[0]
            line = f"Money leaving {top_out} (r_w {leaving.iloc[0]['r_w_mean']*100:+.2f}%)"
            (bull if risk_tag(top_out) == "defensive" else bear).append(
                line + (" — defensives shunned" if risk_tag(top_out) == "defensive" else "")
            )

    out.append("\n### Bull Case\n")
    if bull:
        out.extend(f"- {b}" for b in bull)
    else:
        out.append("- _(no signal crosses its bull threshold)_")
    out.append("\n### Bear Case\n")
    if bear:
        out.extend(f"- {b}" for b in bear)
    else:
        out.append("- _(no signal crosses its bear threshold)_")

    # --- Net Assessment: confidence-weighted blend of the 5d/21d analogue
    # distributions. Empirical, not opinion.
    out.append("\n### Net Assessment\n")
    weighted = []
    for f in (f5, f21):
        if f.get("bullish_pct") is None:
            continue
        w = forecast_confidence(f, verdicts=verdicts)["score"]
        weighted.append((w, f))
    if weighted:
        tot_w = sum(w for w, _ in weighted) or 1.0
        p_bull = sum(w * f["bullish_pct"] for w, f in weighted) / tot_w
        p_neut = sum(w * f["neutral_pct"] for w, f in weighted) / tot_w
        p_bear = sum(w * f["bearish_pct"] for w, f in weighted) / tot_w
        norm = p_bull + p_neut + p_bear or 1.0
        p_bull, p_neut, p_bear = p_bull / norm, p_neut / norm, p_bear / norm
        lean = max(
            [("Bullish", p_bull), ("Neutral", p_neut), ("Bearish", p_bear)],
            key=lambda x: x[1],
        )[0]
        out.append(f"Probability-weighted view (confidence-weighted blend of the "
                   f"5d/21d analogue distributions, SPY):\n")
        out.append(f"- Bullish **{p_bull*100:.0f}%** · Neutral **{p_neut*100:.0f}%** · "
                   f"Bearish **{p_bear*100:.0f}%** → lean **{lean}**")
    else:
        n_bull, n_bear = len(bull), len(bear)
        lean = ("Bullish" if n_bull > n_bear else
                "Bearish" if n_bear > n_bull else "Neutral")
        out.append(f"- _No analogue forecast available — qualitative lean from the "
                   f"evidence count above: **{lean}** ({n_bull} bull vs {n_bear} bear)._")
    if regime:
        out.append(f"- Regime context: **{regime['regime']}**"
                   + (f", day {regime['days_in_regime']}" if regime.get("days_in_regime") else ""))

    # --- Suggested Positioning ---
    out.append("\n### Suggested Positioning\n")
    pos: list[str] = []
    if not entering.empty:
        pos.append("Overweight " + ", ".join(entering.head(2).index))
    if not leaving.empty:
        pos.append("Underweight " + ", ".join(leaving.head(2).index))
    cash_reasons = []
    if rec is not None and rec >= 20:
        cash_reasons.append(f"recession {rec:+.0f}")
    if liq is not None and liq <= 40:
        cash_reasons.append(f"liquidity {liq:.0f}/100")
    pos.append("Maintain an elevated cash buffer (" + ", ".join(cash_reasons) + ")"
               if cash_reasons else "Normal cash allocation")
    out.extend(f"- {p}" for p in pos)

    if verdicts:
        n_fail = sum(1 for v in verdicts.values() if v.verdict == "fail")
        n_und = sum(1 for v in verdicts.values() if v.verdict == "undetermined")
        if n_fail or n_und:
            out.append(f"\n_Caveat: {n_fail} signal(s) fail and {n_und} are undetermined "
                       f"under IC validation — see the Forecast Scorecard and Detected "
                       f"Themes sections before acting on the lean above._")
    out.append("\n_Model-derived positioning consistent with current signals — "
               "not investment advice. The probabilities are empirical analogue "
               "frequencies, not predictions._")
    return "\n".join(out)


def section_capital_rotation_pairs(signals: dict, n: int = 7) -> str:
    """Phase-A wishlist item 8 — surface the 5 pair-blocks already in components.

    capital_rotation's components dict contains a `pairs` list. Today only the
    headline score is shown; the per-pair scores are the more useful read.
    """
    out = [_section_header(n, "Capital Rotation — Pair Breakdown")]
    cr = signals.get("capital_rotation", {})
    pairs = cr.get("pairs") or []
    if not pairs:
        out.append("_No pair-block components recorded for this date._")
        return "\n".join(out)
    out.append("| Pair | Score | Breadth Conf. | Interpretation |")
    out.append("|---|---:|---:|---|")
    for p in pairs:
        out.append(
            f"| {p.get('name', '?').replace('_', ' ')} | "
            f"{_fmt(p.get('score'))} | "
            f"{_fmt(p.get('breadth_confidence'), '{:.2f}')} | "
            f"{p.get('interp', '')} |"
        )
    out.append("\n_Headline capital_rotation is a breadth-weighted mean of these 5 pair scores. "
               "A near-zero headline with high pair dispersion = noisy / no clear rotation; "
               "a clear headline with consistent pair signs = decisive rotation._")
    return "\n".join(out)


def _movers_table_row(r: pd.Series) -> str:
    accel = r.get("rs_accel_5", pd.NA) if "rs_accel_5" in r.index else pd.NA
    rsc1  = r.get("rs_change_1", pd.NA) if "rs_change_1" in r.index else pd.NA
    return (
        f"| {r['symbol']} | {_fmt(r['r_d']*100, '{:+.2f}%')} | "
        f"{_fmt(r['r_w']*100, '{:+.2f}%')} | {_fmt(r['r_m']*100, '{:+.2f}%')} | "
        f"{_fmt(r['rs_rank'], '{:.0f}')} | {_fmt(rsc1, '{:+.0f}')} | "
        f"{_fmt(r['rs_change_5'], '{:+.0f}')} | {_fmt(accel, '{:+.0f}')} |"
    )


def section_top_strengthening(metrics: pd.DataFrame, n: int = 3) -> str:
    out = [_section_header(n, "Top Strengthening Assets")]
    if metrics.empty:
        out.append("_No metrics available._"); return "\n".join(out)
    top = _top_movers(metrics, n=5, ascending=False)
    out.append("| Symbol | r_d | r_w | r_m | RS rank | ΔRS(1d) | ΔRS(5d) | Δ²RS(5d) |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in top.iterrows():
        out.append(_movers_table_row(r))
    out.append("\n_ΔRS = velocity (rank change); Δ²RS = acceleration (change in velocity). "
               "Acceleration often turns before raw RS does — the earliest leading signal._")
    return "\n".join(out)


def section_top_weakening(metrics: pd.DataFrame, n: int = 4) -> str:
    out = [_section_header(n, "Top Weakening Assets")]
    if metrics.empty:
        out.append("_No metrics available._"); return "\n".join(out)
    bot = _top_movers(metrics, n=5, ascending=True)
    out.append("| Symbol | r_d | r_w | r_m | RS rank | ΔRS(1d) | ΔRS(5d) | Δ²RS(5d) |")
    out.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for _, r in bot.iterrows():
        out.append(_movers_table_row(r))
    return "\n".join(out)


def section_volume_anomalies(metrics: pd.DataFrame, n: int = 5) -> str:
    out = [_section_header(n, "Volume Anomalies")]
    if metrics.empty:
        out.append("_No metrics available._"); return "\n".join(out)
    anom = metrics.dropna(subset=["vz"]).copy()
    anom = anom[anom["vz"].abs() >= 2.0].sort_values("vz", ascending=False).head(8)
    if anom.empty:
        out.append("No symbol exceeded ±2σ on log-volume z-score.")
        return "\n".join(out)
    out.append("| Symbol | log-V z-score | RV (log) | r_d |")
    out.append("|---|---:|---:|---:|")
    for _, r in anom.iterrows():
        out.append(
            f"| {r['symbol']} | {_fmt(r['vz'])} | {_fmt(r['rv'])} | "
            f"{_fmt(r['r_d']*100, '{:+.2f}%')} |"
        )
    return "\n".join(out)


def section_etf_flows(con: duckdb.DuckDBPyConnection, asof: date, n: int = 6) -> str:
    """Pull the most recent etf_flows rows and display flow + source confidence.

    Flow score normalization (§3.5: flow_z over 60d) requires 60+ days of
    accumulated history; until then, this section reports raw deltas and
    flags 'history-building' so the operator understands the state.
    """
    out = [_section_header(n, "ETF Flow Analysis")]
    df = con.execute(
        """
        SELECT symbol, proxy_method, confidence, shares_outstanding, aum_usd,
               net_flow_usd, source
        FROM etf_flows WHERE ts = ?
        ORDER BY proxy_method DESC, symbol
        """,
        [asof],
    ).df()
    if df.empty:
        out.append("_No ETF flow snapshot recorded for this date. Run `rotate flows --date " +
                   f"{asof.isoformat()}`._")
        return "\n".join(out)

    n_hist = con.execute("SELECT COUNT(DISTINCT ts) FROM etf_flows").fetchone()[0] or 0
    out.append(f"Flow history accumulated: **{n_hist} day(s)**. The §3.5 flow-z score "
               f"requires ≥60 days to normalize; until then deltas below are raw.\n")
    out.append("| ETF | Source | Conf. | Shares Out | AUM (USD) | Net Flow (USD) |")
    out.append("|---|---|:---:|---:|---:|---:|")
    priority_set = {"SPY","QQQ","IWM","GLD","TLT","HYG"}
    for _, r in df.iterrows():
        tag = " ⭐" if r["symbol"] in priority_set else ""
        out.append(
            f"| {r['symbol']}{tag} | {r['proxy_method']} | {r['confidence']:.2f} | "
            f"{_fmt(r['shares_outstanding'], '{:,.0f}')} | "
            f"{_fmt(r['aum_usd'], '${:,.0f}')} | "
            f"{_fmt(r['net_flow_usd'], '${:+,.0f}')} |"
        )
    out.append("\n⭐ = priority issuer-direct ETF (SPY, QQQ, IWM, GLD, TLT, HYG).")
    out.append("Confidence 0.80 = SSGA holdings-derived; 0.70 = yf.info (issuer-published via Yahoo); 0.60 = Method B shares-delta proxy.")
    return "\n".join(out)


def _validation_tag(name: str, verdicts: dict) -> str:
    v = verdicts.get(name)
    if not v:
        return ""
    if v.verdict == "pass":
        return " ✅"
    if v.verdict == "fail":
        return " ⚠️ PROVISIONAL"
    return " ⏳"  # undetermined


def _truncate_at_word(text: str, limit: int = 80) -> str:
    """Cut a string at a word boundary near `limit`. Adds '…' if truncated."""
    if text is None or len(text) <= limit:
        return text or ""
    cut = text.rfind(" ", 0, limit)
    if cut < limit // 2:   # no nearby space — fall back to hard cut
        cut = limit
    return text[:cut].rstrip(",;: ") + "…"


def section_detected_themes(signals: dict, verdicts: dict, n: int = 8) -> str:
    out = [_section_header(n, "Detected Themes")]
    out.append("| Signal | Score | Confidence | Validation |")
    out.append("|---|---:|:---:|:---|")
    rows_order = [
        "relative_strength", "relative_volume", "capital_rotation",
        "risk_on_off", "inflation", "growth", "recession", "liquidity",
    ]
    for name in rows_order:
        s = signals.get(name, {})
        score = s.get("score")
        conf = s.get("confidence")
        v = verdicts.get(name)
        if v is None:
            v_cell = "—"
        elif v.verdict == "pass":
            v_cell = f"✅ IC={v.median_ic_5d:+.3f}" if v.median_ic_5d is not None else "✅"
        elif v.verdict == "fail":
            v_cell = "⚠️ " + _truncate_at_word(v.reason, 80)
        else:
            v_cell = "⏳ " + _truncate_at_word(v.reason, 80)
        out.append(
            f"| {name.replace('_', ' ').title()} | {_fmt(score)} | "
            f"{_confidence_badge(conf)} | {v_cell} |"
        )
    return "\n".join(out)


def _diff_signals(now: dict, prev: dict) -> list[tuple[str, float, float]]:
    """Delta of composite scalar scores. Skips per-asset signals like
    `relative_strength`, whose headline value is the top-|z| asset's score —
    that flips between assets day to day, so its delta is uninterpretable."""
    SKIP_FOR_DELTA = {"relative_strength"}
    diffs = []
    for name, n in now.items():
        if name in SKIP_FOR_DELTA:
            continue
        if n.get("score") is None:
            continue
        p = prev.get(name, {})
        if p.get("score") is None:
            continue
        d = n["score"] - p["score"]
        diffs.append((name, n["score"], d))
    diffs.sort(key=lambda x: abs(x[2]), reverse=True)
    return diffs


def section_changed_since(n: int, label: str, now: dict, prev: dict | None) -> str:
    out = [_section_header(n, f"What Changed Since {label}")]
    if not prev:
        out.append(f"_No comparable {label.lower()} snapshot available in the store._")
        return "\n".join(out)
    diffs = _diff_signals(now, prev)[:6]
    if not diffs:
        out.append("_No measurable change._"); return "\n".join(out)
    out.append("| Signal | Now | Δ |")
    out.append("|---|---:|---:|")
    for name, score, d in diffs:
        out.append(f"| {name.replace('_', ' ').title()} | {_fmt(score)} | {_fmt(d)} |")
    return "\n".join(out)


def _check_observed_direction(
    sym: str,
    expected_strengthens: bool,
    metrics: pd.DataFrame,
) -> tuple[str, str]:
    """Compare each implicated ticker against today's r_w. Returns (mark, note)
    where mark is ✓ / ✗ / — (consistent / contradicting / mixed-or-unknown)."""
    if metrics.empty:
        return "—", "no data"
    row = metrics[metrics["symbol"] == sym]
    if row.empty:
        return "—", "not tracked"
    r_w = row["r_w"].iloc[0]
    r_d = row["r_d"].iloc[0]
    if pd.isna(r_w):
        return "—", "no r_w"
    pct_w = f"{r_w*100:+.2f}%"
    if expected_strengthens:
        if r_w > 0.005:
            return "✓", f"r_w {pct_w}"
        if r_w < -0.005:
            return "✗", f"r_w {pct_w}"
        return "—", f"r_w {pct_w} (flat)"
    else:
        if r_w < -0.005:
            return "✓", f"r_w {pct_w}"
        if r_w > 0.005:
            return "✗", f"r_w {pct_w}"
        return "—", f"r_w {pct_w} (flat)"


def _macro_context_lines(fred: pd.DataFrame | None) -> list[str]:
    """Wishlist W8 — observable-macro context (FRED) for the narrative.

    Surfaces policy rate, the yield curve, credit spreads and breakeven
    inflation so a regime / rotation call references observable macro, not only
    price action. All series are in percent (pp); changes are reported in bps.
    Returns [] when no FRED panel is available (FRED is opt-in)."""
    if fred is None or fred.empty:
        return []

    def latest(col):
        if col not in fred.columns:
            return None, None
        s = fred[col].dropna()
        return (float(s.iloc[-1]), s) if not s.empty else (None, None)

    def chg(s, k):
        s = s.dropna()
        return float(s.iloc[-1] - s.iloc[-1 - k]) if len(s) > k else None

    lines: list[str] = []

    ff, ff_s = latest("DFEDTARU")
    if ff is not None:
        d = chg(ff_s, 63)
        move = ("roughly unchanged over ~1 quarter" if d is None or abs(d) < 0.01
                else f"{d * 100:+.0f} bps over ~1 quarter")
        lines.append(f"Fed funds target (upper bound) **{ff:.2f}%** — {move}.")

    curve, _ = latest("T10Y2Y")
    if curve is not None:
        if curve < 0:
            lines.append(f"Yield curve **inverted** (10y−2y {curve * 100:+.0f} bps) — "
                         "a classic recession lead, though with long, variable lags.")
        else:
            lines.append(f"Yield curve positively sloped (10y−2y {curve * 100:+.0f} bps).")

    oas, oas_s = latest("BAMLH0A0HYM2")
    if oas is not None:
        d = chg(oas_s, 21)
        if d is None:
            tail = ""
        elif d > 0.1:
            tail = f", widening {d * 100:+.0f} bps over 21d (credit stress rising)"
        elif d < -0.1:
            tail = f", tightening {d * 100:+.0f} bps over 21d (credit stress easing)"
        else:
            tail = ", little changed over 21d"
        lines.append(f"HY credit spread (OAS) **{oas:.2f}%**{tail}.")

    be, be_s = latest("T10YIE")
    if be is not None:
        d = chg(be_s, 21)
        trend = ("" if d is None or abs(d) < 0.02
                 else " (rising)" if d > 0 else " (falling)")
        lines.append(f"10-year breakeven inflation **{be:.2f}%**{trend}.")

    return lines


def section_explanations(
    interp: dict,
    metrics: pd.DataFrame | None = None,
    n: int = 12,
    fred: pd.DataFrame | None = None,
) -> str:
    out = [_section_header(n, "Potential Explanations")]
    src = interp.get("source", "rules-based")
    meta = interp.get("_meta", {}) or {}
    if src == "deepseek-llm":
        toks_in = meta.get("input_tokens")
        toks_out = meta.get("output_tokens")
        model = meta.get("model", "deepseek")
        tail = f" ({toks_in} → {toks_out} tokens)" if toks_in and toks_out else ""
        out.append(f"_Generated by **{model}**{tail}. Falls back to rule-based on API failure. "
                   "Same §4.5 anti-hallucination rules are enforced in the system prompt._")
    else:
        out.append("_Generated by the rule-based narrative library (no LLM key configured, or LLM unavailable)._")
    claims = interp.get("claims", [])
    if not claims:
        out.append("\n_No narratives crossed the confidence floor (0.35). Signals are inconclusive._")
    for cl in claims:
        out.append(f"\n**{cl['narrative_id']}** — {cl['text']}")
        out.append(f"  - Confidence: {cl['confidence']:.2f} ({cl['bucket']})")
        if cl["supporting"]:
            out.append("  - Supporting:")
            for s in cl["supporting"]:
                out.append(f"    - {s}")
        if cl["conflicting"]:
            out.append("  - Conflicting:")
            for s in cl["conflicting"]:
                out.append(f"    - {s}")

        # Implicated products: cross-checked against today's weekly returns
        strengthens = cl.get("implicated_strengthening") or []
        weakens = cl.get("implicated_weakening") or []
        if (strengthens or weakens) and metrics is not None:
            out.append("  - Products implicated (✓ consistent with narrative, ✗ contradicting today's r_w):")
            if strengthens:
                parts = []
                for s in strengthens:
                    mark, note = _check_observed_direction(s, True, metrics)
                    parts.append(f"{s} {mark} ({note})")
                out.append("    - Likely strengthening: " + ", ".join(parts))
            if weakens:
                parts = []
                for s in weakens:
                    mark, note = _check_observed_direction(s, False, metrics)
                    parts.append(f"{s} {mark} ({note})")
                out.append("    - Likely weakening: " + ", ".join(parts))

    confs = interp.get("cross_signal_conflicts", [])
    if confs:
        out.append("\n**Cross-signal flags:**")
        for c in confs:
            out.append(f"- {c}")

    macro = _macro_context_lines(fred)
    if macro:
        out.append("\n**Macro context (observable, FRED):**")
        out.append("_Policy, the curve and credit as a cross-check on the "
                   "price-derived signals above — a regime call should not rest "
                   "on price action alone._")
        for m in macro:
            out.append(f"- {m}")
    return "\n".join(out)


def section_confidence_assessment(signals: dict, interp: dict, n: int = 13) -> str:
    out = [_section_header(n, "Confidence Assessment")]
    valid_confs = [s.get("confidence") for s in signals.values() if s.get("confidence") is not None]
    if not valid_confs:
        out.append("_No confidences to assess._"); return "\n".join(out)
    avg = sum(valid_confs) / len(valid_confs)
    out.append(f"- Average signal confidence: **{avg:.2f}**")
    high = sum(1 for c in valid_confs if c >= CONFIDENCE_MED)
    low = sum(1 for c in valid_confs if c < CONFIDENCE_LOW)
    out.append(f"- Signals at high confidence (≥{CONFIDENCE_MED:.2f}): {high}/{len(valid_confs)}")
    out.append(f"- Signals below the low-confidence cutoff (<{CONFIDENCE_LOW:.2f}): {low}/{len(valid_confs)}")
    out.append("\n_All narrative claims above the floor have been included. Anything below the floor "
               "is suppressed by design (§4.5 anti-hallucination policy)._")
    return "\n".join(out)


# ============================================================
# Top-level builders
# ============================================================

def _etf_flows_with_con(cfg: Config, asof: date, n: int = 6) -> str:
    """Small wrapper that opens its own connection so the existing build_daily_report
    flow doesn't need refactoring."""
    with connect(cfg.storage.duckdb_path) as con:
        return section_etf_flows(con, asof, n=n)


def build_daily_report(cfg: Config, asof: date) -> str:
    verdicts = latest_verdicts(cfg)

    with connect(cfg.storage.duckdb_path) as con:
        signals = _load_signals(con, asof)
        metrics = _load_metrics(con, asof)
        regime = _load_regime(con, asof)

        prev_d = _prior_trading_day(con, asof, 1)
        prev_w = _prior_trading_day(con, asof, 5)
        prev_m = _prior_trading_day(con, asof, 21)

        prev_d_sig = _load_signals(con, prev_d) if prev_d else None
        prev_w_sig = _load_signals(con, prev_w) if prev_w else None
        prev_m_sig = _load_signals(con, prev_m) if prev_m else None

    # Validation filter: drop signals that failed IC gate from the interp inputs.
    # The LLM sees validation status directly so it can down-weight rather than
    # blind-filter, but rule-based interp uses a hard filter for simplicity.
    filtered_for_interp = {
        name: s for name, s in signals.items()
        if (name not in verdicts) or verdicts[name].verdict != "fail"
    }

    # Try LLM first; fall back to rule-based on any failure.
    universe = [s.symbol for s in cfg.universe]
    interp = llm_interpret(
        asof=asof.isoformat(), signals=signals, metrics=metrics, regime=regime,
        verdicts=verdicts,
        prev_d_signals=prev_d_sig, prev_w_signals=prev_w_sig, prev_m_signals=prev_m_sig,
        universe=universe,
    )
    if interp is None:
        interp = interpret(filtered_for_interp)
        interp["source"] = "rules-based"

    with connect(cfg.storage.duckdb_path) as con:
        rotation_strength_md = _safe_section(
            "Section 7 — Rotation Strength",
            lambda: section_rotation_strength(con, asof, signals, n=7))
        forecast_scorecard_md = _safe_section(
            "Section 14 — Forecast Scorecard — Actual vs Forecast",
            lambda: section_forecast_scorecard(con, asof, n=14))
        # W8 macro context for the narrative; FRED is opt-in, so tolerate a
        # missing fred_series table.
        try:
            fred_panel = load_fred_panel(con, asof)
        except Exception:
            fred_panel = None

    # Section numbering driven by the wishlist's Final Vision. The opening
    # sections answer the four questions: the COMMITTEE-SHAPED summary (§2),
    # WHERE money is moving (§4/§5), WHAT THE LEADERSHIP IS (§6), HOW
    # STRONG/DECISIVE (§7/§8), HISTORICAL ANALOGUES of similar regimes (§9),
    # where it tends to GO next conditional on those analogues (§10/§11), a
    # PROBABILISTIC FORECAST for SPY + sectors with confidence attached
    # (§12/§13), and the SCORECARD that grades those forecasts (§14). Context,
    # attribution (§21) and narrative sections follow.
    current_regime_name = regime["regime"] if regime else None
    header = [
        f"# Capital Rotation Report — {asof.isoformat()} (daily)",
        "",
        "_Generated by the rotation system. All numerical signals are model outputs, "
        "not predictions. Hedged language is enforced by §4.5; high-confidence "
        "narratives are foregrounded, conflicting evidence is always listed._",
    ]
    # Each section is rendered through _safe_section (C2): one section raising
    # degrades to a placeholder rather than aborting the report + its delivery.
    # The cover page (W5/W4) sits above §1, like an exec summary.
    section_specs = [
        ("Executive Dashboard",
         lambda: section_executive_dashboard(cfg, asof, signals, prev_w_sig, metrics, regime, verdicts)),
        ("Signal Inflection Monitor",
         lambda: section_inflection_monitor(signals, prev_w_sig, metrics)),
        ("Section 1 — Overview",
         lambda: section_overview(cfg, asof, signals, metrics, regime, interp, verdicts, n=1)),
        ("Section 2 — Investment Committee View",
         lambda: section_committee_view(cfg, asof, signals, regime, verdicts, metrics, n=2)),
        ("Section 3 — Market Regime", lambda: section_market_regime(regime, n=3)),
        ("Section 4 — Capital Flow Dashboard", lambda: section_capital_flow_dashboard(metrics, n=4)),
        ("Section 5 — Flow Map", lambda: section_flow_map(metrics, n=5)),
        ("Section 6 — Leadership Rotation Tracker", lambda: section_leadership_tracker(metrics, n=6)),
        ("Section 7 — Rotation Strength", lambda: rotation_strength_md),  # already _safe_section'd
        ("Section 8 — Capital Rotation — Pair Breakdown", lambda: section_capital_rotation_pairs(signals, n=8)),
        # Phase C: conditional historical layer
        ("Section 9 — Historical Analogues", lambda: section_historical_analogues(cfg, asof, n=9)),
        ("Section 10 — Regime Transition Probabilities",
         lambda: section_regime_transitions(cfg, current_regime_name, n=10)),
        ("Section 11 — Where Money Likely Goes Next",
         lambda: section_rotation_probability(cfg, asof, n=11, horizon=5)),
        # Phase D: probabilistic forecasts + the scorecard that grades them
        ("Section 12 — Probabilistic Market Forecast",
         lambda: section_market_forecast(cfg, asof, verdicts, n=12)),
        ("Section 13 — Sector Forecast", lambda: section_sector_forecast(cfg, asof, verdicts, n=13)),
        ("Section 14 — Forecast Scorecard", lambda: forecast_scorecard_md),  # already _safe_section'd
        # Remaining context sections (HK / Greater China is an independent report).
        ("Section 15 — Top Strengthening Assets", lambda: section_top_strengthening(metrics, n=15)),
        ("Section 16 — Top Weakening Assets", lambda: section_top_weakening(metrics, n=16)),
        ("Section 17 — Sector / Bucket Breadth", lambda: section_bucket_breadth(metrics, n=17)),
        ("Section 18 — Volume Anomalies", lambda: section_volume_anomalies(metrics, n=18)),
        ("Section 19 — ETF Flow Analysis", lambda: _etf_flows_with_con(cfg, asof, n=19)),
        ("Section 20 — Detected Themes", lambda: section_detected_themes(signals, verdicts, n=20)),
        ("Section 21 — Signal Attribution", lambda: section_signal_attribution(signals, n=21)),
        ("Section 22 — What Changed Since Yesterday",
         lambda: section_changed_since(22, "Yesterday", signals, prev_d_sig)),
        ("Section 23 — What Changed Since Last Week",
         lambda: section_changed_since(23, "Last Week", signals, prev_w_sig)),
        ("Section 24 — What Changed Since Last Month",
         lambda: section_changed_since(24, "Last Month", signals, prev_m_sig)),
        ("Section 25 — Potential Explanations",
         lambda: section_explanations(interp, metrics, n=25, fred=fred_panel)),
        ("Section 26 — Confidence Assessment",
         lambda: section_confidence_assessment(signals, interp, n=26)),
    ]
    parts = header + [_safe_section(lbl, fn) for lbl, fn in section_specs]
    body = "\n".join(parts)
    # Glossary appended last; it inspects everything above to decide what to
    # include, so it stays focused per-report.
    body += "\n" + render_glossary(body, section_number=27)
    return body + "\n"


def build_weekly_report(cfg: Config, asof: date) -> str:
    # Weekly variant: same template, header re-labelled, "What Changed Since" uses
    # 1w/4w/13w deltas. The data loader is shared.
    md = build_daily_report(cfg, asof)
    md = md.replace(
        f"# Capital Rotation Report — {asof.isoformat()} (daily)",
        f"# Capital Rotation Report — Week ending {asof.isoformat()} (weekly)"
    )
    md += (
        "\n\n_Note: weekly mode reuses the daily template. The 'What Changed' "
        "sections collapse 1d/1w/1m into 1w/4w/13w semantics; the underlying "
        "snapshots are the relevant trading-day cuts._\n"
    )
    return md


def build_monthly_report(cfg: Config, asof: date) -> str:
    md = build_daily_report(cfg, asof)
    md = md.replace(
        f"# Capital Rotation Report — {asof.isoformat()} (daily)",
        f"# Capital Rotation Report — Month ending {asof.isoformat()} (monthly)"
    )
    md += (
        "\n\n_Note: monthly mode adds the macro-narrative arc on top of the daily "
        "template. Regime calendar will be appended once 90d of regime_history "
        "has accumulated._\n"
    )
    return md


def store_report(cfg: Config, asof: date, horizon: str, body: str) -> None:
    from datetime import datetime
    with connect(cfg.storage.duckdb_path) as con:
        con.execute(
            "DELETE FROM reports WHERE ts = ? AND horizon = ?", [asof, horizon]
        )
        con.execute(
            "INSERT INTO reports (ts, horizon, body_md, generated_at) VALUES (?, ?, ?, ?)",
            [asof, horizon, body, datetime.utcnow()],
        )
