"""Risk Analyzer agent.

Synthesizes:
  - Stock's own price / realized volatility
  - Macro signals (yield curve, VIX, drawdowns, inflation, etc.)
  - Fundamental quality
  - Recent news headlines

…into base/bull/bear scenarios over the configured horizon, plus a
probability distribution over drawdown magnitudes — the exact input the
Tactical Exit policy needs.

The LLM's `drawdown_probabilities` are sanity-checked against a non-LLM
historical-rolling-window bootstrap (`bootstrap_drawdown_probabilities`). When
the LLM diverges from the empirical base rate by more than
BOOTSTRAP_CLAMP_MAX_DIVERGENCE, we clamp toward the prior. This makes the
deterministic tactical ladder less hostage to a single LLM call's drift.
"""
from __future__ import annotations

import logging
from typing import Optional

import numpy as np
import pandas as pd

from src.agents.base import fmt_optional, fmt_pct
from src.data.fundamentals import FundamentalsSnapshot
from src.data.macro import MacroSnapshot
from src.data.prices import PriceSnapshot
from src.llm.client import chat_json
from src.models.schemas import (
    ForwardCatalysts,
    FundamentalAssessment,
    RiskAssessment,
    ValuationResult,
)

log = logging.getLogger(__name__)

# If |LLM P(dd≥X%) − bootstrap P(dd≥X%)| exceeds this, clamp LLM toward the prior.
# 0.20 = 20 percentage points; chosen empirically to allow LLM judgment to dominate
# unless it diverges from base rates implausibly.
BOOTSTRAP_CLAMP_MAX_DIVERGENCE = 0.20

# Minimum number of rolling H-day windows required before the bootstrap is considered
# reliable enough to clamp against. Thin histories return no prior (no clamping).
BOOTSTRAP_MIN_WINDOWS = 30


def bootstrap_drawdown_probabilities(
    history: pd.DataFrame, horizon_days: int
) -> Optional[dict[str, float]]:
    """Empirical P(drawdown ≥ X%) over the next `horizon_days`, from rolling windows.

    For each starting day t in the data, computes the max drawdown observed in the
    H-day window that follows: `1 - min(close[t+1..t+H]) / close[t]`. Returns the
    empirical fraction where this exceeds each of {10, 15, 20, 25}%.

    Returns None if history is too short (< horizon_days + BOOTSTRAP_MIN_WINDOWS).
    """
    if history is None or len(history) == 0 or "Close" not in history.columns:
        return None
    closes = history["Close"].dropna().to_numpy()
    n = len(closes)
    if n < horizon_days + BOOTSTRAP_MIN_WINDOWS:
        return None

    drawdowns: list[float] = []
    for t in range(n - horizon_days):
        window = closes[t + 1 : t + 1 + horizon_days]
        if window.size == 0:
            continue
        dd_pct = (1.0 - float(window.min()) / float(closes[t])) * 100.0
        drawdowns.append(max(0.0, dd_pct))

    if len(drawdowns) < BOOTSTRAP_MIN_WINDOWS:
        return None

    dds = np.array(drawdowns)
    return {
        "10": float((dds >= 10).mean()),
        "15": float((dds >= 15).mean()),
        "20": float((dds >= 20).mean()),
        "25": float((dds >= 25).mean()),
    }


def clamp_against_bootstrap(
    llm: dict[str, float],
    bootstrap: dict[str, float],
    max_divergence: float = BOOTSTRAP_CLAMP_MAX_DIVERGENCE,
) -> tuple[dict[str, float], list[str]]:
    """Clamp each LLM bucket toward the bootstrap when divergence > max_divergence.

    Returns (clamped_probabilities, list_of_human_readable_notes).
    Buckets within tolerance are passed through unchanged.
    """
    out: dict[str, float] = {}
    notes: list[str] = []
    for k in ("10", "15", "20", "25"):
        llm_val = float(llm.get(k, 0.0))
        boot_val = float(bootstrap.get(k, 0.0))
        if abs(llm_val - boot_val) > max_divergence:
            # Clamp to the boundary of the tolerance band, not all the way to the prior.
            clamped = boot_val + max_divergence if llm_val > boot_val else boot_val - max_divergence
            clamped = max(0.0, min(1.0, clamped))
            notes.append(
                f"P(dd≥{k}%): LLM={llm_val:.2f} bootstrap={boot_val:.2f} "
                f"(|Δ|={abs(llm_val - boot_val):.2f}>{max_divergence:.2f}) → clamped to {clamped:.2f}"
            )
            out[k] = clamped
        else:
            out[k] = llm_val
    return out, notes


SYSTEM_PROMPT = """You are a senior buy-side risk analyst.

Your job: produce a FORWARD-LOOKING risk assessment over the requested horizon.

Output structure (must match schema):
  - scenarios: list of named scenarios (typically "base", "bull", "bear"; you may add a 4th if a tail case is material). Probabilities MUST sum to 1.0.
  - drawdown_probabilities: a dict with keys exactly "10", "15", "20", "25" — each value is the probability the stock will draw down by AT LEAST that percent from current levels within the horizon. These should be MONOTONICALLY DECREASING (P(>=10) >= P(>=15) >= P(>=20) >= P(>=25)). Validate this yourself before responding.
  - realized_vol_annualized_pct: copy from input.
  - key_macro_signals: the 3-7 most relevant macro/idiosyncratic signals driving the assessment.
  - horizon_days: copy from input.

Principles:
  - Be honest about uncertainty. Per the investment framework: "do not depend on prediction; depend on survival." Drawdown probabilities should reflect base rates, not bold forecasts.
  - Anchor on realized volatility: a 50%+ annualized vol stock has ~15-20% monthly moves as normal. A 20% vol stock does not.
  - Factor in the macro signals — yield curve inversion, VIX regime, sector-specific risks, geopolitical exposure.
  - If the stock is already deeply drawn down from its 52w high, FURTHER drawdown probabilities may be elevated OR reduced (oversold bounce vs falling-knife) — assess specifically.
  - Calibrate to history: over a 90-day window, a healthy market stock with 25% vol has roughly:
       P(drawdown >=10%) ≈ 30-40%, P(>=15%) ≈ 15-20%, P(>=20%) ≈ 7-12%, P(>=25%) ≈ 3-6%
    Adjust UP for elevated macro stress, weak fundamentals, or high vol.
    Adjust DOWN for strong fundamentals, defensive sectors, and benign macro.
  - Each scenario's rationale should be 1-2 sentences.
"""


def _format_user(
    symbol: str,
    p: PriceSnapshot,
    f: FundamentalsSnapshot,
    fa: FundamentalAssessment,
    v: ValuationResult,
    m: MacroSnapshot,
    horizon_days: int,
    forward_catalysts: Optional[ForwardCatalysts] = None,
) -> str:
    headlines = "\n".join(
        f"  - {h.get('headline', '')}" for h in f.recent_headlines[:6]
    ) or "  (none)"
    macro_signals = "\n".join(f"  - {s}" for s in m.signals) or "  (none firing)"

    if forward_catalysts and forward_catalysts.key_catalysts:
        cat_lines = "\n".join(
            f"  - {c.event} [{c.direction}, conf={c.confidence}, ~{c.expected_magnitude_pct or '?'}%] "
            f"{c.expected_date.isoformat() if c.expected_date else 'no date'} — {c.rationale}"
            for c in forward_catalysts.key_catalysts
        )
        macro_overlay_lines = "\n".join(f"  - {o}" for o in forward_catalysts.macro_overlay) or "  (none)"
        catalyst_section = f"""

FORWARD CATALYSTS (next {forward_catalysts.horizon_days} days, from Information Retrieval agent):
{cat_lines}

MACRO OVERLAY (events that specifically affect this stock):
{macro_overlay_lines}

NEWS SENTIMENT: score={forward_catalysts.sentiment_score:+.2f} — {forward_catalysts.sentiment_summary}
"""
    else:
        catalyst_section = "\n\nFORWARD CATALYSTS: (Information Retrieval agent did not run, or no material catalysts found)\n"

    return f"""SYMBOL: {symbol}
HORIZON: {horizon_days} days
CURRENT PRICE: {p.current_price:.2f} {p.currency}

STOCK CONTEXT:
  Realized vol (annualized, 30d): {p.realized_vol_annualized_pct:.1f}%
  Return 1m / 3m / 1y:             {p.cumulative_return_1m_pct:.1f}% / {p.cumulative_return_3m_pct:.1f}% / {p.cumulative_return_1y_pct:.1f}%
  Sector / Industry:               {f.sector} / {f.industry}

QUALITY SUMMARY:
  Quality score: {fa.quality_score:.1f}/10
  Moat:          {fa.moat_strength}
  Balance sheet: {fa.balance_sheet_health}
  Red flags:     {'; '.join(fa.red_flags) or '(none)'}

VALUATION:
  Intrinsic range:    {v.intrinsic_low:.2f} – {v.intrinsic_base:.2f} – {v.intrinsic_high:.2f} {v.currency}
  Margin of safety:   {v.margin_of_safety_pct:.1f}% (positive = undervalued)
  Confidence:         {v.confidence}

MACRO SNAPSHOT:
  10Y-3M spread:       {fmt_optional(m.yield_curve_10y3m_pct, '{:.2f}%')}
  10Y-2Y spread:       {fmt_optional(m.yield_curve_10y2y_pct, '{:.2f}%')}
  VIX:                 {fmt_optional(m.vix_level)}
  Fed funds rate:      {fmt_optional(m.fed_funds_rate_pct, '{:.2f}%')}
  US Unemployment:     {fmt_optional(m.unemployment_rate_pct, '{:.1f}%')}
  US CPI YoY:          {fmt_optional(m.cpi_yoy_pct, '{:.1f}%')}
  S&P 500 vs 52w high: {fmt_optional(m.sp500_drawdown_pct, '{:.1f}%')}
  Hang Seng vs 52w high: {fmt_optional(m.hsi_drawdown_pct, '{:.1f}%')}

ACTIVE MACRO SIGNALS:
{macro_signals}

RECENT HEADLINES:
{headlines}{catalyst_section}

Produce the risk assessment per the system prompt. Ensure drawdown_probabilities
keys are exactly "10","15","20","25" and the values are monotonically decreasing.
If forward catalysts include high-impact NEGATIVE events with high confidence, increase
drawdown probabilities accordingly. If catalysts skew POSITIVE, you may reduce drawdown
probabilities — but never below sensible base rates (markets always have tail risk).
"""


def analyze_risk(
    symbol: str,
    p: PriceSnapshot,
    f: FundamentalsSnapshot,
    fa: FundamentalAssessment,
    v: ValuationResult,
    m: MacroSnapshot,
    horizon_days: int = 90,
    forward_catalysts: Optional[ForwardCatalysts] = None,
) -> RiskAssessment:
    user_msg = _format_user(symbol, p, f, fa, v, m, horizon_days, forward_catalysts)
    # temperature=0: drawdown_probabilities directly feed the deterministic tactical
    # ladder. Any drift here translates into noise in the discipline layer.
    result = chat_json(
        system=SYSTEM_PROMPT,
        user=user_msg,
        schema=RiskAssessment,
        temperature=0.0,
    )

    # Force-overwrite the realized vol and horizon to match inputs exactly
    result.realized_vol_annualized_pct = p.realized_vol_annualized_pct
    result.horizon_days = horizon_days

    # Bootstrap sanity check — clamp LLM probabilities toward the empirical base rate
    # when they diverge implausibly. Always store the prior on the result so the
    # dashboard can show both LLM and historical numbers side by side.
    bootstrap = bootstrap_drawdown_probabilities(p.history, horizon_days)
    if bootstrap is not None:
        clamped, clamp_notes = clamp_against_bootstrap(result.drawdown_probabilities, bootstrap)
        result.drawdown_probabilities = clamped
        result.bootstrap_drawdown_probabilities = bootstrap
        result.bootstrap_clamp_notes = clamp_notes
        if clamp_notes:
            log.info(
                "Risk %s — drawdown probabilities clamped against bootstrap prior:\n  %s",
                symbol, "\n  ".join(clamp_notes),
            )

    # Enforce monotonicity defensively (must run AFTER clamping)
    keys = ["10", "15", "20", "25"]
    probs = [result.drawdown_probabilities.get(k, 0.0) for k in keys]
    for i in range(1, len(probs)):
        if probs[i] > probs[i - 1]:
            probs[i] = probs[i - 1]
    result.drawdown_probabilities = dict(zip(keys, probs))

    return result
