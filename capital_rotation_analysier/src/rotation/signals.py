"""Composite signal scores per the project docs §3.2 and §3.3.

All signed scores are mapped to [-100, +100] via 100·tanh(S/2).
Magnitude-only scores (Relative Volume, Liquidity) are mapped to [0, 100].

Sign convention: positive = risk-on / inflationary / growth / liquid.

Each signal function returns a dict:
    {
      "score": float | None,        # the headline score
      "confidence": float | None,   # 0..1
      "components": dict[str, ...], # breakdown for the report
    }
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from . import metrics as M


def _tanh_bounded(s: float, scale: float = 2.0) -> float:
    """100·tanh(s/scale). Smooth, bounded, monotonic."""
    return float(100.0 * np.tanh(s / scale))


def _safe_mean(vals: Iterable[float]) -> float | None:
    v = [x for x in vals if x is not None and not np.isnan(x)]
    return float(np.mean(v)) if v else None


def _last_valid(series: pd.Series) -> float | None:
    s = series.dropna()
    return float(s.iloc[-1]) if len(s) else None


def _sign_agreement(values: list[float]) -> float:
    """Fraction agreeing on dominant sign. 1.0 = unanimous, 0.5 = split."""
    v = [x for x in values if x is not None and not np.isnan(x)]
    if not v:
        return 0.0
    pos = sum(1 for x in v if x > 0)
    neg = sum(1 for x in v if x < 0)
    return max(pos, neg) / len(v)


def _attribution(drivers: list[dict], raw: float, baseline: float = 0.0) -> dict:
    """Uniform per-driver breakdown stored alongside each composite score (W3).

    drivers: [{"driver", "value", "weight", "contrib"}, ...] where
    sum(contrib) == `raw`, the pre-mapping aggregate (the tanh/level input).
    `baseline` is the headline score at raw == 0 (50 for the 0..100 liquidity
    level, 0 for signed signals). The report scales each driver into headline
    score points via  pts_i = contrib_i / raw * (score - baseline)  so the
    points column sums exactly to the score.
    """
    return {
        "drivers": sorted(drivers, key=lambda d: abs(d["contrib"]), reverse=True),
        "raw": float(raw),
        "baseline": float(baseline),
    }


# ============================================================
# §3.2.1 Relative Strength Score (per-asset)
# ============================================================

def relative_strength(close: pd.DataFrame, asof: pd.Timestamp) -> dict:
    """Cross-sectional z of blended-horizon return on the asof date.

    Returns one score per asset; the headline `score` is the highest-magnitude RS
    so the alert layer has a single number, but the breakdown is the useful part.
    """
    br = M.blended_return(close)
    if asof not in br.index:
        return {"score": None, "confidence": None, "per_asset": {}, "components": {}}
    row = br.loc[asof].dropna()
    if row.empty:
        return {"score": None, "confidence": None, "per_asset": {}, "components": {}}

    z = M.cross_sectional_z(row).clip(-3, 3)
    per_asset = {sym: _tanh_bounded(z[sym]) for sym in z.index}

    # Headline = top |z|·sign(z), readable as the dominant RS leader/laggard.
    top = z.abs().idxmax()
    headline = per_asset[top]

    # Confidence: do the horizons agree for the top mover?
    horizons = {1: "1d", 5: "5d", 21: "21d", 63: "63d"}
    horizon_signs = []
    for h in horizons:
        r = M.log_returns(close[[top]], h)
        if asof in r.index and not pd.isna(r.loc[asof, top]):
            horizon_signs.append(np.sign(r.loc[asof, top]))
    if horizon_signs:
        conf = sum(1 for s in horizon_signs if s == np.sign(headline)) / len(horizon_signs)
    else:
        conf = 0.0

    return {
        "score": headline,
        "confidence": float(conf),
        "per_asset": per_asset,
        "components": {"top_symbol": top, "horizon_signs": horizon_signs},
    }


# ============================================================
# §3.2.2 Relative Volume Score (0..100, magnitude)
# ============================================================

def relative_volume_score(volume: pd.DataFrame, asof: pd.Timestamp) -> dict:
    rv = M.relative_volume(volume)
    vz = M.volume_zscore(volume)
    if asof not in rv.index:
        return {"score": None, "confidence": None, "components": {}}

    rv_row = rv.loc[asof].dropna()
    vz_row = vz.loc[asof].dropna()
    if rv_row.empty:
        return {"score": None, "confidence": None, "components": {}}

    # Universe-level RV signal = mean activity vs normal
    rv_mean = float(rv_row.mean())
    vz_mean = float(vz_row.mean()) if not vz_row.empty else 0.0

    raw = np.tanh(rv_mean) + float(np.clip(vz_mean / 3.0, -1, 1))
    # 50 * raw maps [-2,+2] -> [-100, 100]; shift to 0..100 magnitude
    score = float(50.0 + 25.0 * raw)
    score = max(0.0, min(100.0, score))

    # Confidence: agreement of signs between rv and vz per asset
    common = list(set(rv_row.index) & set(vz_row.index))
    if common:
        agree = sum(
            1 for s in common if np.sign(rv_row[s]) == np.sign(vz_row[s])
        ) / len(common)
    else:
        agree = 0.0

    return {
        "score": score,
        "confidence": float(agree),
        "components": {"rv_mean": rv_mean, "vz_mean": vz_mean},
    }


# ============================================================
# §3.3 Capital Rotation Score (marquee, signed)
# ============================================================

BLOCK_PAIRS = [
    {
        "name": "equities_vs_bonds",
        "block_a": ["SPY", "QQQ", "IWM"],
        "block_b": ["TLT", "IEF"],
        "interp": "Capital rotating into equity risk away from duration",
    },
    {
        "name": "growth_vs_defensives",
        "block_a": ["QQQ", "SMH", "XLY"],
        "block_b": ["XLP", "XLU", "GLD"],
        "interp": "Cyclical/growth bid over defensives",
    },
    {
        "name": "semis_vs_staples",
        "block_a": ["SMH"],
        "block_b": ["XLP"],
        "interp": "Leading-edge growth bid",
    },
    {
        "name": "us_vs_international",
        "block_a": ["SPY"],
        "block_b": ["EZU", "EWJ", "EEM"],
        "interp": "US outperformance vs international developed/EM",
    },
    {
        "name": "crypto_vs_gold",
        "block_a": ["BTC-USD"],
        "block_b": ["GLD"],
        "interp": "Speculative-liquidity bid over hard-money hedge",
    },
    {
        # Wishlist §16 — China vs US tech (KWEB tracks the same Chinese internet
        # names as US-listed ADRs but at HK valuations). KWEB outperforming QQQ
        # signals capital rotating from US tech to Greater China tech.
        "name": "us_vs_china_tech",
        "block_a": ["QQQ"],
        "block_b": ["KWEB"],
        "interp": "US tech bid over Chinese tech (positive) or the reverse (negative)",
    },
    {
        # Broader US-vs-China cut: SPY against FXI/MCHI. Captures the macro
        # rotation when one bloc decisively outperforms the other.
        "name": "us_vs_china_broad",
        "block_a": ["SPY"],
        "block_b": ["FXI", "MCHI"],
        "interp": "US large-cap bid over Chinese large-cap (positive) or the reverse (negative)",
    },
]


def _block_momentum(close: pd.DataFrame, syms: list[str]) -> pd.Series:
    """Equal-weighted mean of robust-z'd monthly log returns across constituents.

    Returns a Series indexed by date with one momentum value per day.
    """
    present = [s for s in syms if s in close.columns]
    if not present:
        return pd.Series(dtype=float)
    rm = M.log_returns(close[present], 21)
    z = rm.apply(lambda col: M.robust_z(col), axis=0)
    return z.mean(axis=1)


def _breadth_within(close: pd.DataFrame, syms: list[str], asof: pd.Timestamp) -> float:
    """|fraction with positive r_w − 0.5| · 2 → 1.0 if unanimous, 0 if split."""
    present = [s for s in syms if s in close.columns]
    if not present:
        return 0.0
    rw = M.log_returns(close[present], 5)
    if asof not in rw.index:
        return 0.0
    row = rw.loc[asof].dropna()
    if row.empty:
        return 0.0
    frac_pos = float((row > 0).sum() / len(row))
    return abs(frac_pos - 0.5) * 2.0


def capital_rotation(close: pd.DataFrame, asof: pd.Timestamp) -> dict:
    """5-pair block rotation. Headline = breadth-weighted mean of pair scores.

    Per §3.3: the per-pair breakdown is more useful than the headline.
    """
    pair_results = []
    for pair in BLOCK_PAIRS:
        mom_a = _block_momentum(close, pair["block_a"])
        mom_b = _block_momentum(close, pair["block_b"])
        if asof not in mom_a.index or asof not in mom_b.index:
            continue
        a, b = mom_a.loc[asof], mom_b.loc[asof]
        if pd.isna(a) or pd.isna(b):
            continue
        pair_score = _tanh_bounded(float(a - b))
        breadth = min(
            _breadth_within(close, pair["block_a"], asof),
            _breadth_within(close, pair["block_b"], asof),
        )
        pair_results.append({
            "name": pair["name"],
            "score": pair_score,
            "breadth_confidence": float(breadth),
            "interp": pair["interp"],
        })

    if not pair_results:
        return {"score": None, "confidence": None, "pairs": []}

    total_w = sum(p["breadth_confidence"] for p in pair_results) or 1.0
    headline = sum(p["score"] * p["breadth_confidence"] for p in pair_results) / total_w
    # Confidence = average breadth-confidence across pairs
    conf = float(np.mean([p["breadth_confidence"] for p in pair_results]))

    # Headline is already the weighted mean of pair scores (no second tanh),
    # so contributions are in score points directly: raw == score.
    drivers = [
        {
            "driver": p["name"],
            "value": p["score"],
            "weight": p["breadth_confidence"] / total_w,
            "contrib": p["score"] * p["breadth_confidence"] / total_w,
        }
        for p in pair_results
    ]

    return {
        "score": float(headline),
        "confidence": conf,
        "pairs": pair_results,
        "attribution": _attribution(drivers, float(headline)),
    }


# ============================================================
# §3.2.4 Risk-On / Risk-Off Score (signed)
# ============================================================

RISK_ON_BASKET = ["SPY", "QQQ", "IWM", "EEM", "BTC-USD", "HYG"]
RISK_OFF_BASKET = ["TLT", "GLD", "UUP", "FXF"]


def risk_on_off(close: pd.DataFrame, asof: pd.Timestamp) -> dict:
    """Equal-weighted z-diff of risk-on vs risk-off monthly returns.

    Formation window is 21d (monthly), not 5d. The 5d formation was used in the
    original §3.2.4 spec but produced consistently negative rolling-IC against
    forward 5d returns — short-horizon equity returns are mean-reverting at the
    5d-vs-5d frequency, which structurally inverts a "recent momentum" score
    against its own near-future. Switching to 21d formation captures trend, which
    is positively serially correlated, and yields +0.121 / +0.336 median IC at
    5d / 21d horizons on the local 244-obs backfill (vs -0.057 / +0.124 at 5d).
    See design note S1.ROO (2026-06-10) for the per-constituent diagnostic.
    """
    rm = M.log_returns(close, 21)
    if asof not in rm.index:
        return {"score": None, "confidence": None, "components": {}}

    on_syms = [s for s in RISK_ON_BASKET if s in close.columns]
    off_syms = [s for s in RISK_OFF_BASKET if s in close.columns]
    if not on_syms or not off_syms:
        return {"score": None, "confidence": None, "components": {}}

    z = rm.apply(lambda col: M.robust_z(col), axis=0)
    row = z.loc[asof]
    on_vals = row[on_syms].dropna()
    off_vals = row[off_syms].dropna()
    on_z = on_vals.mean()
    off_z = off_vals.mean()
    if pd.isna(on_z) or pd.isna(off_z):
        return {"score": None, "confidence": None, "components": {}}

    score = _tanh_bounded(float(on_z - off_z))

    on_row = rm.loc[asof, on_syms].dropna()
    off_row = rm.loc[asof, off_syms].dropna()
    frac_on_pos = float((on_row > 0).sum() / len(on_row)) if len(on_row) else 0.5
    frac_off_pos = float((off_row > 0).sum() / len(off_row)) if len(off_row) else 0.5
    conf = float(abs(frac_on_pos - frac_off_pos))

    # Equal-weighted within each basket: raw = mean(on z) − mean(off z).
    drivers = [
        {"driver": s, "value": float(on_vals[s]),
         "weight": 1.0 / len(on_vals), "contrib": float(on_vals[s]) / len(on_vals)}
        for s in on_vals.index
    ] + [
        {"driver": s, "value": float(off_vals[s]),
         "weight": -1.0 / len(off_vals), "contrib": -float(off_vals[s]) / len(off_vals)}
        for s in off_vals.index
    ]

    return {
        "score": score,
        "confidence": conf,
        "components": {
            "on_z": float(on_z), "off_z": float(off_z),
            "frac_on_pos": frac_on_pos, "frac_off_pos": frac_off_pos,
            "formation_horizon_days": 21,
        },
        "attribution": _attribution(drivers, float(on_z - off_z)),
    }


# ============================================================
# §3.2.5 Inflation Score (signed)  -- requires copper/oil/gold/silver/TLT
# ============================================================

def inflation_score(
    close: pd.DataFrame,
    asof: pd.Timestamp,
    fred: pd.DataFrame | None = None,
) -> dict:
    """Weighted z-diff of inflation-sensitive vs inflation-vulnerable monthly returns.

    CPER (copper) was previously in the basket at weight +0.25 *and* the IC
    harness's forward target. That target leakage produced consistently negative
    per-component IC for CPER (-0.135/-0.287/-0.418 at 5d/21d/63d on the local
    backfill) — commodities mean-revert at monthly horizons, and including the
    test target in its own predictor structurally inverts the IC. CPER is now
    excluded from the basket; the score still summarises real-economy inflation
    through USO/GLD/SLV/TLT, and the harness target moved to USO (out-of-basket
    until we add TIP). See design note S1.INFL (2026-06-10) for the per-constituent
    diagnostic.

    If FRED breakeven series are available (D1), T10YIE is added at weight +0.30 —
    the market-implied breakeven inflation rate is a direct measure of inflation
    expectations and dominates the noisier commodity proxies when present.
    """
    rm = M.log_returns(close, 21)
    if asof not in rm.index:
        return {"score": None, "confidence": None, "components": {}}

    # Weights from §3.2.5, minus CPER (target leakage — see docstring).
    inputs = {
        "USO": 0.25,    # oil - real-economy inflation
        "GLD": 0.10,    # gold - monetary inflation
        "SLV": 0.10,    # silver - monetary inflation
        "TLT": -0.20,   # bonds inverse: TLT down -> inflation up
    }
    contribs = {}
    weighted_sum = 0.0
    used_weight = 0.0
    for sym, w in inputs.items():
        if sym not in close.columns:
            continue
        z_series = M.robust_z(rm[sym])
        if asof not in z_series.index or pd.isna(z_series.loc[asof]):
            continue
        z = float(z_series.loc[asof])
        contribs[sym] = {"r_m_z": z, "weight": w, "contrib": z * w}
        weighted_sum += z * w
        used_weight += abs(w)

    # FRED T10YIE breakeven inflation. The series is a rate (pct), not a return,
    # so we z-score the 21d change to keep contributors on the same scale.
    if fred is not None and "T10YIE" in fred.columns:
        be = fred["T10YIE"].reindex(close.index).ffill()
        be_chg = be.diff(21)
        z_series = M.robust_z(be_chg)
        if asof in z_series.index and not pd.isna(z_series.loc[asof]):
            w = 0.30
            z = float(z_series.loc[asof])
            contribs["T10YIE_chg21"] = {"r_m_z": z, "weight": w, "contrib": z * w}
            weighted_sum += z * w
            used_weight += abs(w)

    if used_weight < 0.5:
        return {"score": None, "confidence": None, "components": contribs}

    score = _tanh_bounded(weighted_sum)
    signs = [np.sign(c["contrib"]) for c in contribs.values()]
    conf = _sign_agreement(signs)

    drivers = [
        {"driver": k, "value": c["r_m_z"], "weight": c["weight"], "contrib": c["contrib"]}
        for k, c in contribs.items()
    ]

    return {
        "score": float(score),
        "confidence": float(conf),
        "components": {**contribs, "formation_horizon_days": 21},
        "attribution": _attribution(drivers, weighted_sum),
    }


# ============================================================
# §3.2.6 Growth Score (signed)
# ============================================================

GROWTH_LEADERS = ["SMH", "XLI", "IYT", "CPER", "EEM"]
GROWTH_LAGGARDS = ["XLU", "XLP", "TLT"]


def growth_score(close: pd.DataFrame, asof: pd.Timestamp) -> dict:
    rm = M.log_returns(close, 21)
    if asof not in rm.index:
        return {"score": None, "confidence": None, "components": {}}

    lead_present = [s for s in GROWTH_LEADERS if s in close.columns]
    lag_present = [s for s in GROWTH_LAGGARDS if s in close.columns]
    if not lead_present or not lag_present:
        return {"score": None, "confidence": None, "components": {}}

    z = rm.apply(lambda col: M.robust_z(col), axis=0)
    row = z.loc[asof]
    lead_vals = row[lead_present].dropna()
    lag_vals = row[lag_present].dropna()
    lead_z = lead_vals.mean()
    lag_z = lag_vals.mean()
    if pd.isna(lead_z) or pd.isna(lag_z):
        return {"score": None, "confidence": None, "components": {}}

    score = _tanh_bounded(float(lead_z - lag_z))
    lead_row = rm.loc[asof, lead_present].dropna()
    frac_lead_pos = float((lead_row > 0).sum() / len(lead_row)) if len(lead_row) else 0.0

    drivers = [
        {"driver": s, "value": float(lead_vals[s]),
         "weight": 1.0 / len(lead_vals), "contrib": float(lead_vals[s]) / len(lead_vals)}
        for s in lead_vals.index
    ] + [
        {"driver": s, "value": float(lag_vals[s]),
         "weight": -1.0 / len(lag_vals), "contrib": -float(lag_vals[s]) / len(lag_vals)}
        for s in lag_vals.index
    ]

    return {
        "score": score,
        "confidence": float(frac_lead_pos),
        "components": {
            "leaders_z": float(lead_z),
            "laggards_z": float(lag_z),
            "frac_leaders_pos": frac_lead_pos,
        },
        "attribution": _attribution(drivers, float(lead_z - lag_z)),
    }


# ============================================================
# §3.2.7 Recession Concern Score (signed, +ve = more recession risk)
# ============================================================

def recession_concern(
    close: pd.DataFrame,
    asof: pd.Timestamp,
    fred: pd.DataFrame | None = None,
) -> dict:
    """TLT−IEF curve proxy, HYG/LQD credit spread proxy, XLY/XLP, IYT trend, UUP trend.

    Each input is signed *toward* recession (curve inversion+, HYG/LQD falling+,
    XLY/XLP falling+, IYT falling+, UUP rising+). Then weighted sum, tanh-mapped.

    When FRED is wired (D1):
      - Curve proxy upgrades from TLT−IEF returns to the raw T10Y2Y spread
        (inversion-negative). The raw rate is cleaner than the bond-ETF
        performance differential.
      - Credit proxy upgrades from HYG/LQD ratio to the BAMLH0A0HYM2 HY OAS
        (widening = recessionary).
      ETF-based proxies remain as fallbacks when FRED is not available.
    """
    if asof not in close.index:
        return {"score": None, "confidence": None, "components": {}}

    # 21d log returns of the relevant series
    rm = M.log_returns(close, 21)

    components = {}
    contribs = []

    # 1. Curve: prefer FRED T10Y2Y (raw spread); fall back to TLT−IEF returns.
    used_fred_curve = False
    if fred is not None and "T10Y2Y" in fred.columns:
        spread = fred["T10Y2Y"].reindex(close.index).ffill()
        z = M.robust_z(spread)
        if asof in z.index and not pd.isna(z.loc[asof]):
            curve = -float(z.loc[asof])  # inverted/tight spread -> +recession
            components["curve_t10y2y_inv"] = curve
            contribs.append(("curve_t10y2y_inv", curve, 0.25))
            used_fred_curve = True
    if not used_fred_curve and {"TLT", "IEF"}.issubset(close.columns):
        z_tlt = M.robust_z(rm["TLT"]).loc[asof] if asof in rm.index else None
        z_ief = M.robust_z(rm["IEF"]).loc[asof] if asof in rm.index else None
        if z_tlt is not None and z_ief is not None and not pd.isna(z_tlt) and not pd.isna(z_ief):
            curve = float(z_tlt - z_ief)
            components["curve_tlt_minus_ief"] = curve
            contribs.append(("curve_tlt_minus_ief", curve, 0.25))

    # 2. Credit: prefer FRED HY OAS (BAMLH0A0HYM2); fall back to HYG/LQD ratio.
    used_fred_credit = False
    if fred is not None and "BAMLH0A0HYM2" in fred.columns:
        oas = fred["BAMLH0A0HYM2"].reindex(close.index).ffill()
        z = M.robust_z(oas)
        if asof in z.index and not pd.isna(z.loc[asof]):
            credit = float(z.loc[asof])  # widening OAS -> +recession
            components["credit_hy_oas"] = credit
            contribs.append(("credit_hy_oas", credit, 0.25))
            used_fred_credit = True
    if not used_fred_credit and {"HYG", "LQD"}.issubset(close.columns):
        ratio = (close["HYG"] / close["LQD"]).pct_change(21, fill_method=None)
        z = M.robust_z(ratio)
        if asof in z.index and not pd.isna(z.loc[asof]):
            credit = -float(z.loc[asof])  # falling ratio -> +recession
            components["credit_hyg_lqd"] = credit
            contribs.append(("credit_hyg_lqd", credit, 0.25))

    # 3. XLY/XLP ratio: discretionary underperforming staples -> consumer caution -> recession +
    if {"XLY", "XLP"}.issubset(close.columns):
        ratio = (close["XLY"] / close["XLP"]).pct_change(21, fill_method=None)
        z = M.robust_z(ratio)
        if asof in z.index and not pd.isna(z.loc[asof]):
            disc = -float(z.loc[asof])
            components["disc_xly_xlp"] = disc
            contribs.append(("disc_xly_xlp", disc, 0.20))

    # 4. IYT trend: transports falling -> goods slowdown -> recession +
    if "IYT" in close.columns and asof in rm.index:
        z_iyt = M.robust_z(rm["IYT"]).loc[asof]
        if not pd.isna(z_iyt):
            iyt = -float(z_iyt)
            components["transports_iyt"] = iyt
            contribs.append(("transports_iyt", iyt, 0.15))

    # 5. UUP trend: USD rising = liquidity tightening / global stress -> recession +
    if "UUP" in close.columns and asof in rm.index:
        z_uup = M.robust_z(rm["UUP"]).loc[asof]
        if not pd.isna(z_uup):
            usd = float(z_uup)
            components["usd_uup"] = usd
            contribs.append(("usd_uup", usd, 0.15))

    if not contribs:
        return {"score": None, "confidence": None, "components": components}

    raw = sum(v * w for _, v, w in contribs)
    score = _tanh_bounded(raw)

    # Confidence: how many inputs agree on the sign of `raw`?
    signs = [np.sign(v * w) for _, v, w in contribs]
    conf = _sign_agreement(signs)

    drivers = [
        {"driver": name, "value": v, "weight": w, "contrib": v * w}
        for name, v, w in contribs
    ]

    return {
        "score": float(score),
        "confidence": float(conf),
        "components": components,
        "attribution": _attribution(drivers, float(raw)),
    }


# ============================================================
# §3.2.8 Liquidity Score (0..100, magnitude/level)
# ============================================================

def liquidity_score(
    close: pd.DataFrame,
    asof: pd.Timestamp,
    fred: pd.DataFrame | None = None,
) -> dict:
    """Liquidity composite (0..100). All inputs robust-z'd.

    Contributors (per §3.2.8, with the adjustments noted in §1.6.1):
      - UUP (DXY proxy)  — inverted: strong USD = tightening
      - SPY realized vol — inverted: high vol = stressed liquidity
      - BTC monthly return — positive: high-beta liquidity tell
      - ^MOVE bond-vol index — inverted: high MOVE = bond stress (if available)
      - FRED RRPONTSYD — inverted: high RRP = parked cash = tighter (if FRED wired)
      - FRED net Fed liquidity (WALCL − WTREGEN − RRPONTSYD) — positive: more
        liquidity in the system. 21d change is z-scored. When this is available
        we DOWNWEIGHT the standalone RRP contributor (factor 0.5) to avoid
        double-counting (RRP is already a term in the net-liquidity formula).
    """
    if asof not in close.index:
        return {"score": None, "confidence": None, "components": {}}

    components = {}
    contribs = []
    omitted = []

    if "UUP" in close.columns:
        z = M.robust_z(close["UUP"])
        if asof in z.index and not pd.isna(z.loc[asof]):
            v = -float(z.loc[asof])
            components["uup_inv"] = v
            contribs.append(v)

    if "SPY" in close.columns:
        vol = M.realized_vol(close[["SPY"]], 30)["SPY"]
        z = M.robust_z(vol)
        if asof in z.index and not pd.isna(z.loc[asof]):
            v = -float(z.loc[asof])
            components["spy_vol_inv"] = v
            contribs.append(v)

    if "BTC-USD" in close.columns:
        rm = M.log_returns(close[["BTC-USD"]], 21)["BTC-USD"]
        z = M.robust_z(rm)
        if asof in z.index and not pd.isna(z.loc[asof]):
            v = float(z.loc[asof])
            components["btc_r21"] = v
            contribs.append(v)

    # ^MOVE index (yfinance) — bond market vol. High MOVE -> bond stress -> tighter.
    if "^MOVE" in close.columns:
        z = M.robust_z(close["^MOVE"])
        if asof in z.index and not pd.isna(z.loc[asof]):
            v = -float(z.loc[asof])
            components["move_inv"] = v
            contribs.append(v)
    else:
        omitted.append("MOVE_index")

    # FRED net Fed liquidity: WALCL − WTREGEN − RRPONTSYD. 21d change z-scored.
    # When this is present we still include the standalone RRP contributor but
    # at half weight (factor 0.5) — RRP appears in both, so undiscounted use
    # would double-count it.
    have_net_liq = (
        fred is not None
        and {"WALCL", "WTREGEN", "RRPONTSYD"}.issubset(fred.columns)
    )
    if have_net_liq:
        w_a = fred["WALCL"].reindex(close.index).ffill()
        w_t = fred["WTREGEN"].reindex(close.index).ffill()
        w_r = fred["RRPONTSYD"].reindex(close.index).ffill()
        # WALCL/WTREGEN are $M, RRPONTSYD is $B. Convert all to $B.
        net_liq = (w_a / 1000.0) - (w_t / 1000.0) - w_r
        net_chg = net_liq.diff(21)
        z = M.robust_z(net_chg)
        if asof in z.index and not pd.isna(z.loc[asof]):
            v = float(z.loc[asof])
            components["net_fed_liquidity_chg21"] = v
            contribs.append(v)
    else:
        omitted.append("Fed_NetLiquidity")

    # FRED RRP — parked cash at the Fed. High RRP -> tighter liquidity.
    # Half-weight when net-liquidity is also present (avoid double-count).
    if fred is not None and "RRPONTSYD" in fred.columns:
        rrp = fred["RRPONTSYD"].reindex(close.index).ffill()
        z = M.robust_z(rrp)
        if asof in z.index and not pd.isna(z.loc[asof]):
            v = -float(z.loc[asof])
            if have_net_liq:
                v *= 0.5
            components["rrp_inv"] = v
            contribs.append(v)
    else:
        omitted.append("Fed_RRP")

    if not contribs:
        if omitted:
            components["_omitted"] = omitted
        return {"score": None, "confidence": None, "components": components}

    mean_z = float(np.mean(contribs))
    score = float(np.clip(50.0 + 100.0 / 6.0 * mean_z, 0.0, 100.0))
    conf = _sign_agreement([np.sign(c) for c in contribs])

    # Equal-weighted mean of the named contributors (the rrp_inv value already
    # carries its 0.5 double-count discount). Level score → baseline 50.
    n = len(contribs)
    drivers = [
        {"driver": name, "value": v, "weight": 1.0 / n, "contrib": v / n}
        for name, v in components.items()
    ]
    if omitted:
        components["_omitted"] = omitted
    return {
        "score": score,
        "confidence": float(conf),
        "components": components,
        "attribution": _attribution(drivers, mean_z, baseline=50.0),
    }


# ============================================================
# Unified confidence (§3.6) — for any single score
# ============================================================

@dataclass
class ConfidenceInputs:
    freshness: float           # 0..1 (newest tick age)
    source_agreement: float    # 0..1 (sign agreement of inputs)
    horizon_agreement: float   # 0..1 (1d/5d/21d consistent?)
    breadth_support: float     # 0..1 (% constituents with same sign)


def unified_confidence(
    ci: ConfidenceInputs,
    w_f: float = 0.20, w_a: float = 0.30, w_h: float = 0.25, w_b: float = 0.25,
) -> float:
    return float(
        w_f * ci.freshness
        + w_a * ci.source_agreement
        + w_h * ci.horizon_agreement
        + w_b * ci.breadth_support
    )


# ============================================================
# Run-all dispatcher
# ============================================================

SIGNAL_FUNCS = {
    "relative_strength": relative_strength,
    "relative_volume":   relative_volume_score,   # takes volume, not close
    "capital_rotation":  capital_rotation,
    "risk_on_off":       risk_on_off,
    "inflation":         inflation_score,
    "growth":            growth_score,
    "recession":         recession_concern,
    "liquidity":         liquidity_score,
}


def compute_all(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    asof: pd.Timestamp,
    fred: pd.DataFrame | None = None,
) -> dict[str, dict]:
    out: dict[str, dict] = {}
    fred_consumers = {"liquidity", "inflation", "recession"}
    for name, fn in SIGNAL_FUNCS.items():
        try:
            if name == "relative_volume":
                out[name] = fn(volume, asof)
            elif name in fred_consumers:
                out[name] = fn(close, asof, fred=fred)
            else:
                out[name] = fn(close, asof)
        except Exception as exc:  # surface, don't crash the whole run
            out[name] = {"score": None, "confidence": None, "error": str(exc)}
    return out
