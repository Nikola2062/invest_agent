"""Hong Kong / Greater China daily report — independent of the US report,
but with the SAME 27-section structure so the two documents read identically.

HK-listed symbols trade on HKEX (different calendar from NYSE) and are
deliberately excluded from the NYSE-aligned signals panel, so this report
computes everything from raw_bars on the HKEX trading days: r_d/r_w/r_m,
IBD-style RS rank cross-sectional across the HK universe, ΔRS velocity and
acceleration, log-volume anomalies, bucket flows.

Section map (mirrors report.build_daily_report):
  1  Overview                     — HK version
  2  Investment Committee View    — rule-based bull/bear + qualitative lean
  3  Market Regime                — trend heuristic (2800.HK + breadth), NOT the
                                    US signal-based classifier
  4  Capital Flow Dashboard       — shared renderer
  5  Flow Map                     — shared renderer
  6  Leadership Rotation Tracker  — shared renderer
  7  Rotation Strength            — cross-bucket dispersion percentile
  8  Capital Rotation Pairs       — intra-HK pairs + HK-vs-US pairs
  9–10  Analogues / regime trans. — n/a placeholders (need HK signal history)
  11 Where Money Likely Goes Next — W6 heuristic Leadership Persistence forecast
  12 Probabilistic Market Forecast— n/a placeholder (needs HK signal history)
  13 Sector Forecast              — W6 heuristic bucket-level persistence
  14 Forecast Scorecard           — n/a (the W6 heuristic has no grading layer)
  15 Top Strengthening            — shared renderer
  16 Top Weakening                — shared renderer
  17 Sector / Bucket Breadth      — shared renderer
  18 Volume Anomalies             — shared renderer (vz/rv computed here)
  19 ETF Flow Analysis            — HK ETF flows (2800.HK, 2828.HK) via Method B
  20 Detected Themes              — n/a (composite signals are US-panel)
  21 Signal Attribution           — n/a (composite signals are US-panel)
  22–24 What Changed              — RS-rank movers over 1d / 5d / 21d
  25 Potential Explanations       — rule-based hedged observations
  26 Confidence Assessment        — data-coverage statistics
  27 Glossary                     — auto-trimmed
  Appendix: per-ticker detail.
"""
from __future__ import annotations

from datetime import date, timedelta

import duckdb
import numpy as np
import pandas as pd

from .buckets import bucket_for, group_metrics_by_bucket
from .config import Config
from .glossary import render_glossary
from .report import (
    _fmt,
    _safe_section,
    _section_header,
    section_bucket_breadth,
    section_capital_flow_dashboard,
    section_flow_map,
    section_leadership_tracker,
    section_top_strengthening,
    section_top_weakening,
    section_volume_anomalies,
)
from .store import connect

# §3.1 blended-return weights — same recipe as the US RS rank.
_BLEND = {"r_m": 0.4, "r_w": 0.3, "r_q": 0.2, "r_d": 0.1}

_HK_BENCHMARK = "2800.HK"   # Tracker Fund of Hong Kong (HSI ETF)


# ============================================================
# Panel + metrics
# ============================================================

def load_hk_panels(
    con: duckdb.DuckDBPyConnection,
    asof: date,
    lookback_days: int = 420,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Wide adj_close and volume panels (index = HKEX trading day, col = symbol)."""
    df = con.execute(
        """
        SELECT symbol, ts, adj_close, volume
        FROM raw_bars
        WHERE asset_class = 'equity_hk' AND ts <= ? AND ts >= ?
        ORDER BY ts
        """,
        [asof, asof - timedelta(days=lookback_days)],
    ).df()
    if df.empty:
        return pd.DataFrame(), pd.DataFrame()
    df["ts"] = pd.to_datetime(df["ts"])
    close = df.pivot(index="ts", columns="symbol", values="adj_close").sort_index()
    volume = df.pivot(index="ts", columns="symbol", values="volume").sort_index()
    return close, volume


def compute_hk_metrics(
    close: pd.DataFrame,
    volume: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per-symbol metrics row for the latest panel day, shaped exactly like a
    metrics_daily row-set so the US report's section renderers work unchanged.

    Columns: symbol, ts, close, r_d, r_w, r_m, rs_rank, rs_change_1,
    rs_change_5, rs_change_21, rs_accel_5, rv, vz.
    """
    if close.empty:
        return pd.DataFrame()

    logc = np.log(close)
    rets = {
        "r_d": logc.diff(1),
        "r_w": logc.diff(5),
        "r_m": logc.diff(21),
        "r_q": logc.diff(63),
    }
    blended = sum(w * rets[k] for k, w in _BLEND.items())
    # IBD-style 1–99 cross-sectional percentile, daily. With ~15 HK names the
    # rank is coarse — same caveat as the US panel at N≈17 (the project docs §3.1).
    rs = blended.rank(axis=1, pct=True).mul(98).add(1).round()
    rs_change_1 = rs.diff(1)
    rs_change_5 = rs.diff(5)
    rs_change_21 = rs.diff(21)
    rs_accel_5 = rs_change_5 - rs_change_5.shift(5)

    last = close.index[-1]
    out = pd.DataFrame({
        "symbol": close.columns,
        "ts": last,
        "close": close.loc[last].values,
        "r_d": rets["r_d"].loc[last].values,
        "r_w": rets["r_w"].loc[last].values,
        "r_m": rets["r_m"].loc[last].values,
        "rs_rank": rs.loc[last].values,
        "rs_change_1": rs_change_1.loc[last].values,
        "rs_change_5": rs_change_5.loc[last].values,
        "rs_change_21": rs_change_21.loc[last].values,
        "rs_accel_5": rs_accel_5.loc[last].values,
    })

    # Volume metrics per §3.1: RV = ln(V / 30d-median), VZ = z on 60d log-volume.
    rv_last = vz_last = pd.Series(np.nan, index=close.columns)
    if volume is not None and not volume.empty:
        v = volume.reindex(columns=close.columns)
        med30 = v.shift(1).rolling(30, min_periods=20).median()
        with np.errstate(divide="ignore", invalid="ignore"):
            rv = np.log(v / med30)
            logv = np.log(v.where(v > 0))
        mu = logv.rolling(60, min_periods=40).mean()
        sd = logv.rolling(60, min_periods=40).std()
        vz = (logv - mu) / sd
        if last in rv.index:
            rv_last = rv.loc[last]
            vz_last = vz.loc[last]
    out["rv"] = rv_last.values
    out["vz"] = vz_last.values

    return out.dropna(subset=["close"]).reset_index(drop=True)


def _bucket_rw_series(close: pd.DataFrame) -> pd.DataFrame:
    """Per-day bucket-mean weekly return series (columns = buckets)."""
    r_w = np.log(close).diff(5)
    cols = pd.Series({c: bucket_for(c) for c in r_w.columns})
    return r_w.T.groupby(cols).mean().T


# ============================================================
# HK-specific sections (mirroring the US section numbers)
# ============================================================

def _section_hk_overview(
    metrics: pd.DataFrame,
    close: pd.DataFrame,
    hk_asof,
    n: int = 1,
) -> str:
    """HK counterpart of the US Overview. Same 4-block layout, HK-appropriate
    content: regime is the trend heuristic (no composite signals), "likely next"
    is honest n/a (no HK analogue history), confidence is rotation strength +
    breadth + data coverage.
    """
    out = [_section_header(n, "Overview")]
    if metrics.empty:
        out.append("_No HK bars in the store — run a fetch for the HK universe._")
        return "\n".join(out)

    # --- Regime line (HK trend heuristic — mirrors §3) ---
    bench_row = metrics[metrics["symbol"] == _HK_BENCHMARK]
    bench_rm = bench_row.iloc[0]["r_m"] if not bench_row.empty else None
    rw_series = metrics["r_w"].dropna()
    breadth = float((rw_series > 0).sum()) / max(len(rw_series), 1) if not rw_series.empty else None
    if bench_rm is None or pd.isna(bench_rm) or breadth is None:
        state, conf = "Undetermined", "—"
    elif bench_rm > 0.02 and breadth >= 0.6:
        state, conf = "HK Uptrend", "high" if breadth >= 0.75 else "medium"
    elif bench_rm < -0.02 and breadth <= 0.4:
        state, conf = "HK Downtrend", "high" if breadth <= 0.25 else "medium"
    else:
        state, conf = "HK Mixed / Rangebound", "low"
    out.append(
        f"**Regime:** {state} · benchmark ({_HK_BENCHMARK}) r_m "
        f"{_fmt(None if bench_rm is None else bench_rm*100, '{:+.2f}%')} · "
        f"confidence {conf} "
        f"· HKEX session {pd.Timestamp(hk_asof).date().isoformat()}"
    )

    # --- Narrative-first: rule-based top observation (no LLM for HK yet) ---
    obs = _hk_top_observations(metrics)
    out.append("")
    if obs:
        out.append(f"> **Top observation (rules):** {obs[0]}")
        if len(obs) > 1:
            out.append(f"> **Secondary:** {obs[1]}")
    else:
        out.append("> _HK moves are within normal ranges this session; no rotation "
                   "observation crosses the reporting threshold._")

    # --- Where money is leaving / entering ---
    g = group_metrics_by_bucket(metrics).sort_values("r_w_mean", ascending=False)
    THRESH = 0.005  # 50 bps — same as the US report's flat-band
    entering = g[g["r_w_mean"] >  THRESH].head(5)
    leaving  = g[g["r_w_mean"] < -THRESH].sort_values("r_w_mean", ascending=True).head(5)

    out.append("\n### Where money is leaving (week to date)")
    if leaving.empty:
        out.append("\n_No HK bucket below the −50 bps threshold this week._")
    else:
        out.append("\n| Bucket | r_w | Members |")
        out.append("|---|---:|---|")
        for name, r in leaving.iterrows():
            out.append(f"| **{name}** | {r['r_w_mean']*100:+.2f}% | {r['members']} |")

    out.append("\n### Where money is entering (week to date)")
    if entering.empty:
        out.append("\n_No HK bucket above the +50 bps threshold this week._")
    else:
        out.append("\n| Bucket | r_w | Members |")
        out.append("|---|---:|---|")
        for name, r in entering.iterrows():
            out.append(f"| **{name}** | {r['r_w_mean']*100:+.2f}% | {r['members']} |")

    # --- Where money is likely to go next (n/a for HK) ---
    out.append("\n### Where money is likely to go next")
    out.append(
        "\n_n/a — HK analogue engine requires ≥1 year of HK composite-signal "
        "history, which doesn't exist yet (HK has no native risk-on/off, "
        "rotation, growth, etc. composites). Strategic item; see the design notes._"
    )

    # --- How confident (rotation strength + breadth + data coverage) ---
    out.append("\n### How confident")
    conf_lines: list[str] = []

    # Rotation strength = percentile of cross-bucket r_w dispersion vs trailing year.
    disp = _bucket_rw_series(close).std(axis=1).dropna() if not close.empty else pd.Series(dtype=float)
    if len(disp) >= 30:
        today = float(disp.iloc[-1])
        hist = disp.iloc[:-1]
        pctile = float((hist <= today).mean()) * 100
        trail5 = float(disp.iloc[-6:-1].mean()) if len(disp) > 6 else today
        trend = "**rising**" if today > trail5 else "**falling**"
        conf_lines.append(
            f"- **Rotation Strength:** {pctile:.0f}/100 (percentile of HK cross-bucket "
            f"r_w dispersion vs trailing {len(hist)}d) · {trend}"
        )
    elif close is not None and not close.empty:
        conf_lines.append("- **Rotation Strength:** insufficient HK history (<30 sessions)")

    if breadth is not None:
        conf_lines.append(
            f"- **Universe breadth (weekly):** {breadth*100:.0f}% of "
            f"{len(rw_series)} HK names advancing"
        )

    if close is not None and not close.empty:
        names_today = int(close.loc[close.index[-1]].notna().sum())
        names_total = close.shape[1]
        conf_lines.append(
            f"- **Data coverage:** {names_today}/{names_total} HK names "
            f"have a bar on the latest HKEX session"
        )

    if conf_lines:
        out.extend(conf_lines)
    else:
        out.append("- _No HK confidence inputs available._")

    out.append("\n_Numbers above are model outputs, not predictions. Full detail in the sections that follow._\n")
    return "\n".join(out)


def _hk_top_observations(metrics: pd.DataFrame) -> list[str]:
    """Two-line rule-based summary of HK bucket flow + leadership rotation."""
    if metrics.empty:
        return []
    obs: list[str] = []
    g = group_metrics_by_bucket(metrics).sort_values("r_w_mean")
    if not g.empty:
        worst, best = g.iloc[0], g.iloc[-1]
        if worst["r_w_mean"] < -0.005 and best["r_w_mean"] > 0.005:
            obs.append(
                f"The data is consistent with capital rotating out of "
                f"**{g.index[0]}** ({worst['r_w_mean']*100:+.2f}% r_w) into "
                f"**{g.index[-1]}** ({best['r_w_mean']*100:+.2f}% r_w)."
            )
        elif worst["r_w_mean"] < -0.005:
            obs.append(
                f"The data is consistent with money leaving **{g.index[0]}** "
                f"({worst['r_w_mean']*100:+.2f}% r_w)."
            )
        elif best["r_w_mean"] > 0.005:
            obs.append(
                f"The data is consistent with money entering **{g.index[-1]}** "
                f"({best['r_w_mean']*100:+.2f}% r_w)."
            )

    m = metrics.dropna(subset=["rs_rank", "rs_change_5"])
    fading = m[(m["rs_rank"] >= 60) & (m["rs_change_5"] <= -15)]
    emerging = m[(m["rs_rank"] >= 40) & (m["rs_rank"] < 80) & (m["rs_change_5"] >= 15)]
    if not fading.empty and not emerging.empty:
        obs.append(
            f"Leadership rotation in progress: {', '.join(sorted(fading['symbol']))} "
            f"bleeding rank while {', '.join(sorted(emerging['symbol']))} gain."
        )
    return obs


def _section_hk_committee(
    con: duckdb.DuckDBPyConnection,
    metrics: pd.DataFrame,
    asof: date,
    n: int = 2,
) -> str:
    """HK counterpart of the US Investment Committee View (W2).

    Bull/Bear bullets from what HK actually computes (benchmark trend, breadth,
    bucket flows, Southbound flows). The Net Assessment is a qualitative lean —
    HK has no analogue forecast layer yet, so publishing probabilities here
    would be invention. Probabilities arrive with the HK forecast layer (W6)."""
    out = [_section_header(n, "Investment Committee View")]
    if metrics.empty:
        out.append("_No HK metrics available._")
        return "\n".join(out)

    bull: list[str] = []
    bear: list[str] = []

    bench_row = metrics[metrics["symbol"] == _HK_BENCHMARK]
    bench_rm = bench_row.iloc[0]["r_m"] if not bench_row.empty else None
    if bench_rm is not None and not pd.isna(bench_rm):
        if bench_rm > 0.02:
            bull.append(f"Benchmark uptrend ({_HK_BENCHMARK} r_m {bench_rm*100:+.2f}%)")
        elif bench_rm < -0.02:
            bear.append(f"Benchmark downtrend ({_HK_BENCHMARK} r_m {bench_rm*100:+.2f}%)")

    rw = metrics["r_w"].dropna()
    breadth = float((rw > 0).sum()) / max(len(rw), 1) if not rw.empty else None
    if breadth is not None:
        if breadth >= 0.60:
            bull.append(f"Broad participation ({breadth*100:.0f}% of HK names advancing on the week)")
        elif breadth <= 0.40:
            bear.append(f"Narrow market ({breadth*100:.0f}% of HK names advancing on the week)")

    g = group_metrics_by_bucket(metrics).sort_values("r_w_mean", ascending=False)
    entering = g[g["r_w_mean"] > 0.005]
    leaving = g[g["r_w_mean"] < -0.005].sort_values("r_w_mean", ascending=True)
    if not entering.empty:
        bull.append(f"{entering.index[0]} attracting capital "
                    f"(r_w {entering.iloc[0]['r_w_mean']*100:+.2f}%)")
    if not leaving.empty:
        bear.append(f"Money leaving {leaving.index[0]} "
                    f"(r_w {leaving.iloc[0]['r_w_mean']*100:+.2f}%)")

    # Southbound 5-session net — the most leveraged HK flow datapoint we have.
    sb_net = None
    try:
        sb = con.execute(
            "SELECT net_buy_cny_100m FROM stock_connect_flows "
            "WHERE direction = 'southbound' AND ts <= ? ORDER BY ts DESC LIMIT 5",
            [asof],
        ).df()
        if not sb.empty:
            sb_net = float(sb["net_buy_cny_100m"].sum())
    except duckdb.CatalogException:
        pass
    if sb_net is not None and abs(sb_net) >= 10.0:  # ≥10亿 over 5 sessions
        if sb_net > 0:
            bull.append(f"Mainland Southbound net buying ({sb_net:+.0f}亿 CNY over 5 sessions)")
        else:
            bear.append(f"Mainland Southbound net selling ({sb_net:+.0f}亿 CNY over 5 sessions)")

    out.append("\n### Bull Case\n")
    if bull:
        out.extend(f"- {b}" for b in bull)
    else:
        out.append("- _(no input crosses its bull threshold)_")
    out.append("\n### Bear Case\n")
    if bear:
        out.extend(f"- {b}" for b in bear)
    else:
        out.append("- _(no input crosses its bear threshold)_")

    out.append("\n### Net Assessment\n")
    n_bull, n_bear = len(bull), len(bear)
    lean = ("Bullish" if n_bull > n_bear else
            "Bearish" if n_bear > n_bull else "Neutral / Mixed")
    out.append(f"- Qualitative lean: **{lean}** ({n_bull} bull vs {n_bear} bear inputs). "
               f"_No probability weighting — that requires the HK forecast layer "
               f"(§9–§14 are n/a until it ships)._")

    out.append("\n### Suggested Positioning\n")
    pos: list[str] = []
    if not entering.empty:
        pos.append("Overweight " + ", ".join(entering.head(2).index))
    if not leaving.empty:
        pos.append("Underweight " + ", ".join(leaving.head(2).index))
    if not pos:
        pos.append("No bucket exceeds the ±50 bps threshold — no positioning tilt")
    out.extend(f"- {p}" for p in pos)
    out.append("\n_Model-derived positioning consistent with current HK metrics — "
               "not investment advice._")
    return "\n".join(out)


def _section_hk_market_regime(metrics: pd.DataFrame, close: pd.DataFrame, n: int = 2) -> str:
    """Lightweight HK market state. This is a TREND HEURISTIC (benchmark r_m +
    universe breadth), not the US report's signal-median regime classifier —
    HK has no composite-signal history to classify on."""
    out = [_section_header(n, "Market Regime")]
    if metrics.empty:
        out.append("_No HK metrics available._")
        return "\n".join(out)

    bench_rm = None
    bench_row = metrics[metrics["symbol"] == _HK_BENCHMARK]
    if not bench_row.empty:
        bench_rm = bench_row.iloc[0]["r_m"]
    rw = metrics["r_w"].dropna()
    breadth = float((rw > 0).sum()) / max(len(rw), 1) if not rw.empty else None

    if bench_rm is None or pd.isna(bench_rm) or breadth is None:
        state, conf = "Undetermined", "—"
    elif bench_rm > 0.02 and breadth >= 0.6:
        state, conf = "HK Uptrend", "high" if breadth >= 0.75 else "medium"
    elif bench_rm < -0.02 and breadth <= 0.4:
        state, conf = "HK Downtrend", "high" if breadth <= 0.25 else "medium"
    else:
        state, conf = "HK Mixed / Rangebound", "low"

    out.append(f"**Current:** {state}")
    out.append(f"**Confidence:** {conf}")
    out.append(f"\n- Benchmark ({_HK_BENCHMARK}, Hang Seng tracker) r_m: "
               f"{_fmt(None if bench_rm is None else bench_rm*100, '{:+.2f}%')}")
    if breadth is not None:
        out.append(f"- Universe weekly breadth: {breadth*100:.0f}% advancing")
    out.append("\n_Trend heuristic (benchmark monthly return + breadth), not the "
               "US report's signal-median regime classifier — HK composite "
               "signals are not computed yet._")
    return "\n".join(out)


def _section_hk_rotation_strength(metrics: pd.DataFrame, close: pd.DataFrame, n: int = 6) -> str:
    """How unusual is today's cross-bucket dispersion vs the trailing year?
    Mirrors the US 'Rotation Strength' framing with an HK-computable stand-in:
    high dispersion across HK buckets = capital actively re-sorting."""
    out = [_section_header(n, "Rotation Strength")]
    if close.empty:
        out.append("_No HK panel available._")
        return "\n".join(out)

    disp = _bucket_rw_series(close).std(axis=1).dropna()
    if len(disp) < 30:
        out.append("_Insufficient history to score rotation strength (need ≥30 sessions)._")
        return "\n".join(out)

    today = float(disp.iloc[-1])
    hist = disp.iloc[:-1]
    pctile = float((hist <= today).mean()) * 100
    trail5 = float(disp.iloc[-6:-1].mean()) if len(disp) > 6 else today
    trend = "rising" if today > trail5 else "falling"

    out.append(f"- **Rotation Strength:** {pctile:.0f}/100 "
               f"(percentile of cross-bucket r_w dispersion vs trailing {len(hist)} sessions)")
    out.append(f"- **Dispersion today:** {today*100:.2f}pp across HK buckets")
    out.append(f"- **Trend** vs trailing 5d mean: **{trend}**")
    g = group_metrics_by_bucket(metrics).sort_values("r_w_mean", ascending=False)
    if not g.empty:
        out.append(f"- **Widest spread:** {g.index[0]} "
                   f"({_fmt(g.iloc[0]['r_w_mean']*100, '{:+.2f}%')}) vs {g.index[-1]} "
                   f"({_fmt(g.iloc[-1]['r_w_mean']*100, '{:+.2f}%')})")
    out.append("\n_Proxy measure: the US report scores the capital_rotation "
               "composite; HK has no composite signals, so this scores bucket "
               "dispersion — high values mean capital is actively re-sorting "
               "within the HK universe._")
    return "\n".join(out)


# Intra-HK rotation pairs + the wishlist §16 HK-vs-US monitor, one table.
_HK_PAIRS = [
    ("HK Internet", "HK Financials", "hk", "Growth/platform bid vs value/SOE banks"),
    ("HK Internet", "HK Energy",     "hk", "New-economy bid vs old-economy SOEs"),
    ("HK Financials", "HK Energy",   "hk", "Banks/insurers vs oil SOEs"),
    ("HK Internet",  "Technology",   "us", "HK internet platforms vs US tech"),
    ("HK Internet",  "China Internet", "us", "HKEX listings vs US-listed KWEB proxy"),
    ("HK Broad",     "US Large Cap", "us", "Hang Seng trackers vs SPY/QQQ"),
    ("HK Financials", "Financials",  "us", "HK banks/insurers vs XLF"),
]


def _section_hk_rotation_pairs(cfg: Config, metrics_hk: pd.DataFrame, asof: date, n: int = 7) -> str:
    out = [_section_header(n, "Capital Rotation — Pair Breakdown")]
    if metrics_hk.empty:
        out.append("_No HK metrics available._")
        return "\n".join(out)

    g_hk = group_metrics_by_bucket(metrics_hk)
    us_ts = None
    g_us = pd.DataFrame()
    with connect(cfg.storage.duckdb_path) as con:
        us_ts = con.execute(
            "SELECT MAX(ts) FROM metrics_daily WHERE ts <= ?", [asof]
        ).fetchone()[0]
        if us_ts is not None:
            g_us = group_metrics_by_bucket(
                con.execute("SELECT * FROM metrics_daily WHERE ts = ?", [us_ts]).df()
            )

    us_note = (f"US side as of {pd.Timestamp(us_ts).date().isoformat()} (last NYSE "
               f"day with metrics)." if us_ts is not None else
               "US metrics unavailable — cross-market rows show —.")
    out.append(f"_Score = weekly-return spread (left − right) in percentage points. "
               f"{us_note} Positive = left side outperforming._\n")
    out.append("| Pair | Left r_w | Right r_w | Spread | Interpretation |")
    out.append("|---|---:|---:|---:|---|")
    for left, right, side, note in _HK_PAIRS:
        lv = g_hk["r_w_mean"].get(left)
        rv = (g_hk if side == "hk" else g_us)["r_w_mean"].get(right) \
            if (side == "hk" or not g_us.empty) else None
        spread = (lv - rv) if (pd.notna(lv) and rv is not None and pd.notna(rv)) else None
        out.append(
            f"| {left} vs {right} | {_fmt(None if lv is None else lv*100, '{:+.2f}%')} | "
            f"{_fmt(None if rv is None else rv*100, '{:+.2f}%')} | "
            f"{_fmt(None if spread is None else spread*100, '{:+.2f}pp')} | {note} |"
        )
    out.append("\n_Raw r_w spreads, not z-scored pair blocks like the US "
               "capital_rotation signal — read the sign and relative magnitude, "
               "not the absolute level. A persistently positive HK Internet − US "
               "Technology spread is the classic 'rotation into Chinese tech' tell._")
    return "\n".join(out)


def _section_hk_etf_flows(con: duckdb.DuckDBPyConnection, cfg: Config,
                           asof: date, n: int = 19) -> str:
    """HK ETF Flow Analysis + Stock Connect Southbound flows (D2/D5).

    Two blocks in one section:
      1. HK index ETFs (2800.HK Tracker Fund, 2828.HK HSCEI ETF) via the same
         Method B shares-delta proxy as the US panel.
      2. Stock Connect Southbound aggregate flows (Mainland → HK) from
         akshare/Eastmoney. Net buying or selling pressure on the HKEX-listed
         universe from Mainland investors — the single most leveraged daily
         capital-flow datapoint for HK.

    Northbound aggregate flows (HK → A-shares) were discontinued at source on
    2024-08-16 by Chinese authorities; we keep historical rows in the DB but
    don't surface them in the daily report."""
    out = [_section_header(n, "ETF Flow Analysis")]
    out.append("### HK Index ETFs\n")
    hk_etfs = ("2800.HK", "2828.HK")
    placeholders = ",".join(["?"] * len(hk_etfs))
    df = con.execute(
        f"""
        SELECT symbol, proxy_method, confidence, shares_outstanding, aum_usd,
               net_flow_usd, source
        FROM etf_flows WHERE ts = ? AND symbol IN ({placeholders})
        ORDER BY symbol
        """,
        [asof, *hk_etfs],
    ).df()
    if df.empty:
        out.append(
            "_No HK ETF flow snapshot recorded for this date yet. "
            "Once the daily pipeline includes 2800.HK / 2828.HK in its flows "
            "snapshot (D5 wired 2026-06-10), the table below will populate._"
        )
    else:
        n_hist = con.execute(
            f"SELECT COUNT(DISTINCT ts) FROM etf_flows WHERE symbol IN ({placeholders})",
            list(hk_etfs),
        ).fetchone()[0] or 0
        out.append(f"Flow history accumulated: **{n_hist} day(s)**. Same 60-day "
                   f"flow-z normalization as the US panel; until then deltas are raw.\n")
        out.append("| ETF | Source | Conf. | Shares Out | AUM (HKD≈) | Net Flow (HKD≈) |")
        out.append("|---|---|:---:|---:|---:|---:|")
        for _, r in df.iterrows():
            out.append(
                f"| {r['symbol']} | {r['proxy_method']} | {r['confidence']:.2f} | "
                f"{_fmt(r['shares_outstanding'], '{:,.0f}')} | "
                f"{_fmt(r['aum_usd'], '${:,.0f}')} | "
                f"{_fmt(r['net_flow_usd'], '${:+,.0f}')} |"
            )
        out.append("\nAUM / Net Flow are reported in the close-quote currency "
                   "(HKD for HK-listed ETFs). The `aum_usd` column name is legacy — "
                   "the value is `shares_outstanding × close_quote_currency`.")
    out.append(_stock_connect_block(con, asof))
    out.append(_sb_holdings_block(con, asof, [s.symbol for s in cfg.universe
                                              if s.asset_class == "equity_hk"]))
    out.append(_china_policy_block(con, asof))
    return "\n".join(out)


def _china_policy_block(con: duckdb.DuckDBPyConnection, asof: date) -> str:
    """PBOC policy + funding rates (D4). Latest values for RRR, LPR 1Y/5Y, and
    daily SHIBOR 3M/1Y. The interesting fields are: where current rates sit,
    when PBOC last moved, and how SHIBOR has drifted vs the last LPR fix."""
    from .ingest.china_macro_adapter import latest_value
    out = ["\n### China Policy & Liquidity\n"]
    rrr = latest_value(con, "RRR_LARGE_BANKS", asof)
    lpr1y = latest_value(con, "LPR_1Y", asof)
    lpr5y = latest_value(con, "LPR_5Y", asof)
    sb3m = latest_value(con, "SHIBOR_3M", asof)
    sb1y = latest_value(con, "SHIBOR_1Y", asof)

    if not any([rrr, lpr1y, lpr5y, sb3m, sb1y]):
        out.append("_No China macro data yet. Once "
                   "`china_macro_adapter.fetch_and_store_all` has run, the "
                   "PBOC policy + SHIBOR table will populate here._")
        return "\n".join(out)

    out.append("| Indicator | Latest | As of | Notes |")
    out.append("|---|---:|---|---|")
    if rrr:
        mag = rrr.get("magnitude")
        mag_str = f"{mag:+.2f}pp move" if mag is not None else "—"
        out.append(f"| **RRR (large banks)** | {rrr['value']:.2f}% | "
                   f"{rrr['ts'].isoformat()} | last change: {mag_str} |")
    if lpr1y:
        out.append(f"| **LPR 1Y** | {lpr1y['value']:.2f}% | "
                   f"{lpr1y['ts'].isoformat()} | PBOC policy-rate anchor |")
    if lpr5y:
        out.append(f"| **LPR 5Y** | {lpr5y['value']:.2f}% | "
                   f"{lpr5y['ts'].isoformat()} | mortgage anchor |")
    if sb3m:
        out.append(f"| **SHIBOR 3M** | {sb3m['value']:.3f}% | "
                   f"{sb3m['ts'].isoformat()} | interbank funding (daily) |")
    if sb1y:
        out.append(f"| **SHIBOR 1Y** | {sb1y['value']:.3f}% | "
                   f"{sb1y['ts'].isoformat()} | term funding |")

    # Quick read on the SHIBOR-vs-LPR spread (transmission gauge): wide negative
    # spread = market funding well below policy rate = ample liquidity.
    if sb3m and lpr1y:
        spread = sb3m["value"] - lpr1y["value"]
        sign = "ample" if spread < -0.5 else ("tight" if spread > 0 else "neutral")
        out.append(f"\n_SHIBOR 3M − LPR 1Y spread: **{spread:+.2f}pp** ({sign} "
                   f"funding regime). Negative spreads mean market rates are "
                   f"easier than the policy anchor — typically risk-on for HK._")
    return "\n".join(out)


def _sb_holdings_block(con: duckdb.DuckDBPyConnection, asof: date,
                       hk_symbols: list[str]) -> str:
    """Per-stock Southbound holdings — which HK names Mainland investors are
    accumulating vs distributing right now (D3). Pulled from
    stock_connect_holdings; uses the 5-day market-value delta as the ranking.
    Limited to our 15-name HK universe so it stays focused."""
    from .ingest.stock_connect_adapter import load_holdings_for_universe
    out = ["\n### Top Southbound Holders — Universe (5-day Δ market value)\n"]
    df = load_holdings_for_universe(con, hk_symbols, asof, lookback_days=14)
    if df.empty:
        out.append(
            "_No Southbound per-stock holdings data yet. Once "
            "`stock_connect_adapter.refresh_holdings` has run, this table will "
            "show which HK names Mainland investors are accumulating._"
        )
        return "\n".join(out)

    # Use the most recent date in the data (may lag asof by 1 day)
    latest_ts = df["ts"].max()
    snap = df[df["ts"] == latest_ts].copy()
    snap = snap.dropna(subset=["mv_chg_5d_hkd"])
    if snap.empty:
        out.append("_No SB-holding 5-day deltas available for this universe._")
        return "\n".join(out)

    snap = snap.sort_values("mv_chg_5d_hkd", ascending=False)
    out.append(f"_As of {pd.Timestamp(latest_ts).date().isoformat()}. "
               f"Δ values in HKD. Positive = Mainland investors added to position "
               f"over the trailing 5 sessions._\n")
    out.append("| Rank | Symbol | Name | Held (HKD) | Δ 1d | Δ 5d | Δ 10d |")
    out.append("|---|---|---|---:|---:|---:|---:|")
    # Top 3 buyers, top 3 sellers
    head = snap.head(3)
    tail = snap.tail(3).iloc[::-1]
    rank_label = lambda i, side: f"#{i+1} {side}"  # noqa: E731
    for i, (_, r) in enumerate(head.iterrows()):
        out.append(
            f"| {rank_label(i, 'buy')} | {r['symbol_dot_hk']} | {r['name']} | "
            f"{_fmt(r['market_value_hkd'], '${:,.0f}')} | "
            f"{_fmt(r['mv_chg_1d_hkd'], '${:+,.0f}')} | "
            f"{_fmt(r['mv_chg_5d_hkd'], '${:+,.0f}')} | "
            f"{_fmt(r['mv_chg_10d_hkd'], '${:+,.0f}')} |"
        )
    for i, (_, r) in enumerate(tail.iterrows()):
        out.append(
            f"| {rank_label(i, 'sell')} | {r['symbol_dot_hk']} | {r['name']} | "
            f"{_fmt(r['market_value_hkd'], '${:,.0f}')} | "
            f"{_fmt(r['mv_chg_1d_hkd'], '${:+,.0f}')} | "
            f"{_fmt(r['mv_chg_5d_hkd'], '${:+,.0f}')} | "
            f"{_fmt(r['mv_chg_10d_hkd'], '${:+,.0f}')} |"
        )
    out.append(f"\n_{len(snap)} of {len(hk_symbols)} HK universe names have SB-holding "
               f"rows — names missing are either ETFs (Tracker/HSCEI) or H-share "
               f"secondary listings not in Stock Connect._")
    return "\n".join(out)


def _stock_connect_block(con: duckdb.DuckDBPyConnection, asof: date) -> str:
    """Southbound aggregate Stock Connect flows (Mainland → HK), as a subsection
    of §19. Pulled from stock_connect_flows (populated by stock_connect_adapter,
    D2). Values are in 100M CNY (亿). Shows the last 7 sessions and a w-t-d
    rolling sum."""
    out = ["\n### Stock Connect — Southbound (Mainland → HK)\n"]
    try:
        df = con.execute(
            """
            SELECT ts, net_buy_cny_100m, hist_cum_cny_100m, holding_value_cny
            FROM stock_connect_flows
            WHERE direction = 'southbound' AND ts <= ?
            ORDER BY ts DESC LIMIT 7
            """,
            [asof],
        ).df()
    except duckdb.CatalogException:
        df = pd.DataFrame()  # stock_connect_flows table not created yet
    if df.empty:
        out.append(
            "_No Southbound flow data yet. Once `stock_connect_adapter.fetch_and_store_all` "
            "has run (akshare/Eastmoney source), the last 7 sessions will surface here._"
        )
        return "\n".join(out)
    # Order ascending for the table, then descending for the most-recent-first read.
    df = df.sort_values("ts")
    last_5 = df.tail(5)
    wtd_5d = float(last_5["net_buy_cny_100m"].sum())
    wtd_direction = "buying" if wtd_5d > 0 else "selling"
    most_recent_ts = df["ts"].iloc[-1]
    out.append(f"**5-session net** ({last_5['ts'].iloc[0].date().isoformat()} → "
               f"{last_5['ts'].iloc[-1].date().isoformat()}): "
               f"**{wtd_5d:+.1f}亿 CNY** — net Mainland **{wtd_direction}** of HK stocks.\n")
    out.append("| Session | Net Buy (亿 CNY) | Hist Cum (兆 CNY) |")
    out.append("|---|---:|---:|")
    for _, r in df.iloc[::-1].iterrows():
        out.append(
            f"| {pd.Timestamp(r['ts']).date().isoformat()} | "
            f"{_fmt(r['net_buy_cny_100m'], '{:+,.2f}')} | "
            f"{_fmt(r['hist_cum_cny_100m'], '{:,.3f}')} |"
        )
    out.append("\n_亿 = 100M CNY. Positive = Mainland investors net buyers of "
               "HK equities through Stock Connect Southbound; negative = net "
               "sellers. Northbound (HK → A-shares) net flows are no longer "
               "published daily by Chinese authorities (frozen 2024-08-16)._")
    if pd.Timestamp(most_recent_ts).date() != asof:
        out.append(f"\n_Latest available: {pd.Timestamp(most_recent_ts).date().isoformat()} "
                   f"(report asof {asof.isoformat()})._")
    return "\n".join(out)


# ============================================================
# Wishlist W6 — heuristic HK Relative Rotation Forecast
# Fills the §11 / §13 "where next" slots that were N/A, without the analogue
# engine: a Leadership Persistence Score from RS (level), ΔRS (velocity) and
# Δ²RS (acceleration). Momentum and acceleration tend to persist, so a
# strong-and-still-accelerating name is the most likely continued leader. A
# modest, clearly-labelled heuristic beats five N/A sections.
# ============================================================

# Current strength dominates; velocity next; acceleration is the tie-breaker.
_PERSISTENCE_WEIGHTS = {"rs_rank": 0.5, "rs_change_5": 0.3, "rs_accel_5": 0.2}


def _leadership_persistence(metrics: pd.DataFrame) -> pd.DataFrame:
    """Rank HK names by a Leadership Persistence Score.

    Each of RS rank, ΔRS(5d) and Δ²RS(5d) is z-scored cross-sectionally across
    the HK universe, then weighted (0.5 / 0.3 / 0.2). Returns the input frame
    plus a `persistence` column, sorted descending. Rows missing any input
    (short history) are dropped."""
    cols = list(_PERSISTENCE_WEIGHTS)
    if metrics.empty or any(c not in metrics.columns for c in cols):
        out = metrics.copy()
        out["persistence"] = pd.Series(dtype=float)
        return out
    m = metrics.dropna(subset=cols).copy()
    if m.empty:
        return m.assign(persistence=pd.Series(dtype=float))
    score = pd.Series(0.0, index=m.index)
    for col, w in _PERSISTENCE_WEIGHTS.items():
        s = m[col].astype(float)
        std = s.std(ddof=0)
        z = (s - s.mean()) / std if std > 0 else pd.Series(0.0, index=m.index)
        score = score + w * z
    m["persistence"] = score
    return m.sort_values("persistence", ascending=False)


def _persistence_row(r: pd.Series) -> str:
    return (f"| {r['symbol']} | {bucket_for(r['symbol'])} | "
            f"{_fmt(r['persistence'], '{:+.2f}')} | {_fmt(r['rs_rank'], '{:.0f}')} | "
            f"{_fmt(r['rs_change_5'], '{:+.0f}')} | {_fmt(r['rs_accel_5'], '{:+.0f}')} |")


def _section_hk_rotation_forecast(metrics: pd.DataFrame, n: int = 11, top_k: int = 5) -> str:
    """Wishlist W6 — fills the §11 "Where Money Likely Goes Next" N/A slot with a
    per-name heuristic forecast (Likely Leaders / Likely Laggards, next ~20D)."""
    out = [_section_header(n, "Where Money Likely Goes Next")]
    out.append("_**Heuristic** Leadership Persistence forecast (next ~20 HKEX "
               "sessions) — **not** the analogue engine. Names ranked by a "
               "weighted blend of RS rank (level), ΔRS (5d velocity) and Δ²RS "
               "(5d acceleration), each z-scored across the HK universe. "
               "Momentum and acceleration tend to persist; treat as a leaning, "
               "not a prediction._\n")
    ranked = _leadership_persistence(metrics)
    if len(ranked) < 2:
        out.append("_Insufficient RS history across the HK universe for a "
                   "persistence ranking (needs ΔRS and Δ²RS — ≥10 sessions)._")
        return "\n".join(out)
    leaders = ranked.head(top_k)
    laggards = ranked.tail(min(top_k, len(ranked) - len(leaders))).sort_values("persistence")

    out.append("### Likely Leaders (Next 20 Sessions)\n")
    out.append("| Asset | Bucket | Persistence | RS | ΔRS(5d) | Δ²RS(5d) |")
    out.append("|---|---|---:|---:|---:|---:|")
    out.extend(_persistence_row(r) for _, r in leaders.iterrows())

    out.append("\n### Likely Laggards (Next 20 Sessions)\n")
    out.append("| Asset | Bucket | Persistence | RS | ΔRS(5d) | Δ²RS(5d) |")
    out.append("|---|---|---:|---:|---:|---:|")
    out.extend(_persistence_row(r) for _, r in laggards.iterrows())

    out.append("\n_Persistence = 0.5·z(RS) + 0.3·z(ΔRS) + 0.2·z(Δ²RS). A heuristic "
               "leaning toward continuation; it has **no** forward-return validation "
               "yet — grading it needs the HK forecast-scorecard layer (§14)._")
    return "\n".join(out)


def _section_hk_sector_forecast(metrics: pd.DataFrame, n: int = 13, top_k: int = 3) -> str:
    """Wishlist W6 — bucket-level companion to §11: mean persistence per bucket,
    filling the §13 "Sector Forecast — Expected Leaders" N/A slot."""
    out = [_section_header(n, "Sector Forecast — Expected Leaders (Next 21 Days)")]
    out.append("_**Heuristic** bucket-level companion to §11: the mean Leadership "
               "Persistence Score per bucket. Not the analogue engine._\n")
    ranked = _leadership_persistence(metrics)
    if ranked.empty:
        out.append("_Insufficient RS history for a bucket persistence ranking._")
        return "\n".join(out)
    ranked = ranked.copy()
    ranked["bucket"] = ranked["symbol"].map(bucket_for)
    g = ranked.groupby("bucket").agg(
        persistence=("persistence", "mean"),
        members=("symbol", lambda s: ", ".join(sorted(s))),
    ).sort_values("persistence", ascending=False)
    out.append("| Rank | Bucket | Members | Mean Persistence |")
    out.append("|---|---|---|---:|")
    lead = g.head(top_k)
    for i, (name, r) in enumerate(lead.iterrows(), 1):
        out.append(f"| Lead {i} | **{name}** | {r['members']} | "
                   f"{_fmt(r['persistence'], '{:+.2f}')} |")
    lag = g.tail(min(top_k, max(len(g) - len(lead), 0))).sort_values("persistence")
    for i, (name, r) in enumerate(lag.iterrows(), 1):
        out.append(f"| Lag {i} | {name} | {r['members']} | "
                   f"{_fmt(r['persistence'], '{:+.2f}')} |")
    out.append("\n_Same heuristic and caveat as §11 — a continuation leaning, "
               "unvalidated against realized forward returns._")
    return "\n".join(out)


def _hk_regime_state(metrics: pd.DataFrame) -> tuple[str, str, float | None, float | None]:
    """Shared HK trend-regime heuristic (benchmark r_m + breadth), as used in §1
    and §3. Returns (state, confidence, bench_rm, breadth)."""
    if metrics.empty:
        return "Undetermined", "—", None, None
    bench_row = metrics[metrics["symbol"] == _HK_BENCHMARK]
    bench_rm = bench_row.iloc[0]["r_m"] if not bench_row.empty else None
    rw_series = metrics["r_w"].dropna()
    breadth = (float((rw_series > 0).sum()) / max(len(rw_series), 1)
               if not rw_series.empty else None)
    if bench_rm is None or pd.isna(bench_rm) or breadth is None:
        return "Undetermined", "—", bench_rm, breadth
    if bench_rm > 0.02 and breadth >= 0.6:
        return "HK Uptrend", ("high" if breadth >= 0.75 else "medium"), bench_rm, breadth
    if bench_rm < -0.02 and breadth <= 0.4:
        return "HK Downtrend", ("high" if breadth <= 0.25 else "medium"), bench_rm, breadth
    return "HK Mixed / Rangebound", "low", bench_rm, breadth


def _section_hk_executive_dashboard(metrics: pd.DataFrame, close: pd.DataFrame, hk_asof) -> str:
    """Wishlist W5 (HK) — the cover-page decision table. HK has no composite or
    analogue layer, so the rows are RS- and flow-based; the forward read is the
    §11 heuristic Leadership Persistence ranking rather than an analogue lean."""
    out = ["\n## Executive Dashboard\n",
           "_The 5-second read for the HK / Greater China book. Every row is "
           "detailed in a numbered section below — decision summary, not new "
           "information. HK has no composite-signal or analogue layer, so the "
           "rows are RS- and flow-based._\n"]
    if metrics.empty:
        out.append("_No HK bars in the store — run a fetch for the HK universe._")
        return "\n".join(out)

    state, conf, _bench_rm, breadth = _hk_regime_state(metrics)
    g = group_metrics_by_bucket(metrics).sort_values("r_w_mean", ascending=False)
    strongest = weakest = "—"
    flow = "Mixed / no clear rotation"
    if not g.empty:
        strongest = f"{g.index[0]} ({g.iloc[0]['r_w_mean']*100:+.2f}%)"
        weakest = f"{g.index[-1]} ({g.iloc[-1]['r_w_mean']*100:+.2f}%)"
        if g.iloc[0]["r_w_mean"] > 0.005:
            tag = g.iloc[0]["risk_tag"]
            flow = ("Defensive bid" if tag == "defensive"
                    else "Risk-on bid" if tag == "risk_on"
                    else "Mixed / no clear rotation")

    ranked = _leadership_persistence(metrics)
    if len(ranked) >= 2:
        lead, lag = ranked.iloc[0], ranked.iloc[-1]
        likely_leader = f"{lead['symbol']} ({bucket_for(lead['symbol'])})"
        likely_laggard = f"{lag['symbol']} ({bucket_for(lag['symbol'])})"
    else:
        likely_leader = likely_laggard = "—"

    breadth_str = f"{breadth*100:.0f}% advancing" if breadth is not None else "—"
    rows = [
        ("Current Market State", f"{state} (conf. {conf})"),
        ("Capital Flow Direction", flow),
        ("Strongest Sector", strongest),
        ("Weakest Sector", weakest),
        ("Likely Leader (20D, heuristic)", likely_leader),
        ("Likely Laggard (20D, heuristic)", likely_laggard),
        ("Universe Breadth (weekly)", breadth_str),
        ("HKEX Session", pd.Timestamp(hk_asof).date().isoformat()),
    ]
    out.append("| Question | Answer |")
    out.append("|---|---|")
    for q, a in rows:
        out.append(f"| {q} | {a} |")
    out.append("\n_Not investment advice. Likely Leader / Laggard are the §11 "
               "heuristic Leadership Persistence extremes — a continuation leaning, "
               "not a validated forecast._")
    return "\n".join(out)


def _section_hk_inflection_monitor(metrics: pd.DataFrame) -> str:
    """Wishlist W4 (HK) — what's *turning*. HK has no composite signals, so this
    tracks relative-strength momentum: the fastest accelerating / decelerating
    names by Δ²RS, which turns before raw RS does."""
    out = ["\n## Signal Inflection Monitor\n",
           "_What is *changing* in HK leadership. HK has no composite signals, so "
           "this tracks RS momentum: the fastest accelerating / decelerating names "
           "by Δ²RS (5-session acceleration of relative strength)._\n"]
    if metrics.empty or "rs_accel_5" not in metrics.columns:
        out.append("_No HK RS-acceleration data for this session._")
        return "\n".join(out)
    acc = metrics.dropna(subset=["rs_accel_5"])
    if acc.empty:
        out.append("_Insufficient RS history for acceleration (needs ≥10 sessions)._")
        return "\n".join(out)

    rc5 = metrics["rs_change_5"].dropna()
    if not rc5.empty:
        n_up = int((rc5 > 5).sum())
        n_dn = int((rc5 < -5).sum())
        out.append(f"**Net 5d RS momentum:** {n_up} names gaining rank, {n_dn} "
                   f"losing (of {len(rc5)} with history).\n")

    top = acc.sort_values("rs_accel_5", ascending=False).head(5)
    bot = acc.sort_values("rs_accel_5", ascending=True).head(5)
    out.append("| Direction | Asset | Bucket | RS | ΔRS(5d) | Δ²RS(5d) |")
    out.append("|---|---|---|---:|---:|---:|")
    for _, r in top.iterrows():
        out.append(f"| ⇈ accel | {r['symbol']} | {bucket_for(r['symbol'])} | "
                   f"{_fmt(r['rs_rank'], '{:.0f}')} | {_fmt(r['rs_change_5'], '{:+.0f}')} | "
                   f"{_fmt(r['rs_accel_5'], '{:+.0f}')} |")
    for _, r in bot.iterrows():
        out.append(f"| ⇊ decel | {r['symbol']} | {bucket_for(r['symbol'])} | "
                   f"{_fmt(r['rs_rank'], '{:.0f}')} | {_fmt(r['rs_change_5'], '{:+.0f}')} | "
                   f"{_fmt(r['rs_accel_5'], '{:+.0f}')} |")
    out.append("\n_Δ²RS = change in RS velocity. A name with high RS but sharply "
               "negative Δ²RS is a leader losing steam — a turn the static RS rank "
               "has not shown yet._")
    return "\n".join(out)


def _section_na(n: int, title: str, reason: str) -> str:
    return "\n".join([_section_header(n, title), f"_{reason}_"])


_NA_SIGNALS = ("Requires HK-native composite-signal history, which is not computed "
               "yet — the 8 composite signals, regime history and analogue engine "
               "are NYSE-panel concepts. See the same section in the US report; "
               "HK-native signals are tracked as a strategic item in the design notes.")


def _section_hk_changed_since(metrics: pd.DataFrame, n: int, label: str, col: str) -> str:
    """US sections 19–21 show composite-signal deltas; HK has no composites, so
    the same slots show the RS-rank movers over the matching horizon."""
    out = [_section_header(n, f"What Changed Since {label}")]
    if metrics.empty or col not in metrics.columns:
        out.append("_No HK metrics available._")
        return "\n".join(out)
    m = metrics.dropna(subset=[col, "rs_rank"]).copy()
    if m.empty:
        out.append("_Insufficient RS history for this horizon._")
        return "\n".join(out)
    movers = pd.concat([
        m.sort_values(col, ascending=False).head(3),
        m.sort_values(col, ascending=True).head(3),
    ]).drop_duplicates(subset="symbol").sort_values(col, ascending=False)
    out.append("| Symbol | Bucket | RS now | ΔRS | r_w |")
    out.append("|---|---|---:|---:|---:|")
    for _, r in movers.iterrows():
        out.append(
            f"| {r['symbol']} | {bucket_for(r['symbol'])} | "
            f"{_fmt(r['rs_rank'], '{:.0f}')} | {_fmt(r[col], '{:+.0f}')} | "
            f"{_fmt(r['r_w']*100, '{:+.2f}%')} |"
        )
    out.append("\n_RS-rank movers over this horizon (the US report shows "
               "composite-signal deltas here; HK composites are not computed)._")
    return "\n".join(out)


def _section_hk_explanations(metrics: pd.DataFrame, cfg: Config, asof: date, n: int = 25) -> str:
    """Rule-based hedged observations from bucket flows and leadership — the
    HK counterpart of the US report's LLM/rule narrative section."""
    out = [_section_header(n, "Potential Explanations")]
    if metrics.empty:
        out.append("_No HK metrics available._")
        return "\n".join(out)
    out.append("_Rule-based observations (no LLM pass for HK yet). Hedged by design._\n")

    obs: list[str] = []
    g = group_metrics_by_bucket(metrics).sort_values("r_w_mean")
    if not g.empty:
        worst = g.iloc[0]
        if worst["r_w_mean"] < -0.005:
            obs.append(
                f"The data is consistent with money leaving **{g.index[0]}** "
                f"(r_w {worst['r_w_mean']*100:+.2f}%, breadth "
                f"{worst['pct_adv_w']*100:.0f}% advancing)."
            )
        best = g.iloc[-1]
        if best["r_w_mean"] > 0.005:
            obs.append(
                f"The data is consistent with money entering **{g.index[-1]}** "
                f"(r_w {best['r_w_mean']*100:+.2f}%, breadth "
                f"{best['pct_adv_w']*100:.0f}% advancing)."
            )

    m = metrics.dropna(subset=["rs_rank", "rs_change_5"])
    fading = m[(m["rs_rank"] >= 60) & (m["rs_change_5"] <= -15)]
    emerging = m[(m["rs_rank"] >= 40) & (m["rs_rank"] < 80) & (m["rs_change_5"] >= 15)]
    if not fading.empty and not emerging.empty:
        obs.append(
            "Leadership rotation appears underway: "
            f"{', '.join(sorted(fading['symbol']))} are bleeding rank while "
            f"{', '.join(sorted(emerging['symbol']))} gain — consistent with a "
            f"shift from {', '.join(sorted(set(fading['symbol'].map(bucket_for))))} "
            f"toward {', '.join(sorted(set(emerging['symbol'].map(bucket_for))))}."
        )

    if not obs:
        obs.append("HK moves are within normal ranges this session; no rotation "
                   "observation crosses the reporting threshold.")
    out.extend(f"- {o}" for o in obs)
    return "\n".join(out)


def _section_hk_confidence(metrics: pd.DataFrame, close: pd.DataFrame, asof: date, n: int = 26) -> str:
    """Data-coverage statistics — the HK counterpart of the US signal-confidence
    section (HK has no composite-signal confidences to average)."""
    out = [_section_header(n, "Confidence Assessment")]
    if metrics.empty or close.empty:
        out.append("_No HK data to assess._")
        return "\n".join(out)
    hk_asof = close.index[-1].date()
    gap = (asof - hk_asof).days
    depth = close.notna().sum()
    out.append(f"- Names with a bar on the latest HKEX session: "
               f"**{int(close.loc[close.index[-1]].notna().sum())}/{close.shape[1]}**")
    out.append(f"- Median history depth: **{int(depth.median())} sessions** "
               f"(min {int(depth.min())}, RS rank needs ≥63 for the full blend)")
    out.append(f"- Session freshness: HKEX {hk_asof.isoformat()} vs report date "
               f"{asof.isoformat()} ({gap} calendar day(s) gap)")
    out.append("\n_The US report averages composite-signal confidences here; HK "
               "composites are not computed, so this section reports data "
               "coverage instead — the floor on which every section above stands._")
    return "\n".join(out)


def _appendix_ticker_detail(metrics: pd.DataFrame) -> str:
    out = ["\n## Appendix — Per-Ticker Detail\n"]
    if metrics.empty:
        out.append("_No HK bars available._")
        return "\n".join(out)
    rdf = metrics.copy()
    rdf["bucket"] = rdf["symbol"].map(bucket_for)
    rdf = rdf.sort_values(["bucket", "symbol"])
    out.append("| Ticker | Bucket | Last close (HKD) | r_d | r_w | r_m | RS | ΔRS(5d) |")
    out.append("|---|---|---:|---:|---:|---:|---:|---:|")
    for _, r in rdf.iterrows():
        out.append(
            f"| {r['symbol']} | {r['bucket']} | {_fmt(r['close'], '{:.2f}')} | "
            f"{_fmt(r['r_d']*100, '{:+.2f}%')} | {_fmt(r['r_w']*100, '{:+.2f}%')} | "
            f"{_fmt(r['r_m']*100, '{:+.2f}%')} | {_fmt(r['rs_rank'], '{:.0f}')} | "
            f"{_fmt(r['rs_change_5'], '{:+.0f}')} |"
        )
    out.append("\n_Returns are computed on each ticker's own HKEX trading days; "
               "RS rank is cross-sectional across the HK universe only._")
    return "\n".join(out)


# ============================================================
# Builder
# ============================================================

def build_hk_daily_report(cfg: Config, asof: date) -> str:
    """Self-contained HK/Greater-China daily report, section-aligned with the
    US report (same numbering, same titles)."""
    with connect(cfg.storage.duckdb_path) as con:
        close, volume = load_hk_panels(con, asof)
        metrics = compute_hk_metrics(close, volume)
        # §2 and §19 need the connection — build those section strings inside
        # the `with` block so the connection is still open. Wrapped so a query
        # fault there can't sink the whole report (C2).
        committee_section = _safe_section(
            "Section 2 — Investment Committee View",
            lambda: _section_hk_committee(con, metrics, asof, n=2))
        etf_flows_section = _safe_section(
            "Section 19 — ETF Flow Analysis",
            lambda: _section_hk_etf_flows(con, cfg, asof, n=19))

    hk_asof = close.index[-1] if not close.empty else asof

    header = [
        f"# Hong Kong Rotation Report — {asof.isoformat()} (daily)",
        "",
        "_Greater China / HKEX-listed universe. Independent of the US report "
        "but section-aligned with it: same numbering, same titles. Sections "
        "that require composite-signal history are marked n/a — those remain "
        "US-panel concepts for now._",
    ]
    # Each section through _safe_section (C2): one failure degrades to a
    # placeholder, not an aborted report. Cover page (W5/W4) sits above §1.
    section_specs = [
        ("Executive Dashboard", lambda: _section_hk_executive_dashboard(metrics, close, hk_asof)),
        ("Signal Inflection Monitor", lambda: _section_hk_inflection_monitor(metrics)),
        ("Section 1 — Overview", lambda: _section_hk_overview(metrics, close, hk_asof, n=1)),
        ("Section 2 — Investment Committee View", lambda: committee_section),  # already _safe_section'd
        ("Section 3 — Market Regime", lambda: _section_hk_market_regime(metrics, close, n=3)),
        ("Section 4 — Capital Flow Dashboard", lambda: section_capital_flow_dashboard(metrics, n=4)),
        ("Section 5 — Flow Map", lambda: section_flow_map(metrics, n=5)),
        ("Section 6 — Leadership Rotation Tracker", lambda: section_leadership_tracker(metrics, n=6)),
        ("Section 7 — Rotation Strength", lambda: _section_hk_rotation_strength(metrics, close, n=7)),
        ("Section 8 — Capital Rotation — Pair Breakdown", lambda: _section_hk_rotation_pairs(cfg, metrics, asof, n=8)),
        ("Section 9 — Historical Analogues", lambda: _section_na(9, "Historical Analogues", _NA_SIGNALS)),
        ("Section 10 — Regime Transition Probabilities",
         lambda: _section_na(10, "Regime Transition Probabilities", _NA_SIGNALS)),
        ("Section 11 — Where Money Likely Goes Next", lambda: _section_hk_rotation_forecast(metrics, n=11)),
        ("Section 12 — Probabilistic Market Forecast",
         lambda: _section_na(12, "Probabilistic Market Forecast", _NA_SIGNALS)),
        ("Section 13 — Sector Forecast", lambda: _section_hk_sector_forecast(metrics, n=13)),
        ("Section 14 — Forecast Scorecard",
         lambda: _section_na(14, "Forecast Scorecard — Actual vs Forecast",
                             "The scorecard grades published forecasts once their horizon "
                             "elapses. The HK report's only forecast is the heuristic "
                             "Leadership Persistence ranking (§11/§13, wishlist W6), which "
                             "has no realized-return grading layer yet — so there is "
                             "nothing to score here.")),
        ("Section 15 — Top Strengthening Assets", lambda: section_top_strengthening(metrics, n=15)),
        ("Section 16 — Top Weakening Assets", lambda: section_top_weakening(metrics, n=16)),
        ("Section 17 — Sector / Bucket Breadth", lambda: section_bucket_breadth(metrics, n=17)),
        ("Section 18 — Volume Anomalies", lambda: section_volume_anomalies(metrics, n=18)),
        ("Section 19 — ETF Flow Analysis", lambda: etf_flows_section),  # already _safe_section'd
        ("Section 20 — Detected Themes", lambda: _section_na(20, "Detected Themes", _NA_SIGNALS)),
        ("Section 21 — Signal Attribution", lambda: _section_na(21, "Signal Attribution", _NA_SIGNALS)),
        ("Section 22 — What Changed Since Yesterday",
         lambda: _section_hk_changed_since(metrics, 22, "Yesterday", "rs_change_1")),
        ("Section 23 — What Changed Since Last Week",
         lambda: _section_hk_changed_since(metrics, 23, "Last Week", "rs_change_5")),
        ("Section 24 — What Changed Since Last Month",
         lambda: _section_hk_changed_since(metrics, 24, "Last Month", "rs_change_21")),
        ("Section 25 — Potential Explanations", lambda: _section_hk_explanations(metrics, cfg, asof, n=25)),
        ("Section 26 — Confidence Assessment", lambda: _section_hk_confidence(metrics, close, asof, n=26)),
    ]
    parts = header + [_safe_section(lbl, fn) for lbl, fn in section_specs]
    body = "\n".join(parts)
    # Glossary is Section 27 in both reports; the per-ticker appendix follows
    # it but is included in the glossary's term scan so appendix-only tickers
    # still get their entries.
    appendix = _appendix_ticker_detail(metrics)
    body += "\n" + render_glossary(body + "\n" + appendix, section_number=27)
    body += "\n" + appendix
    return body + "\n"
