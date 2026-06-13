"""Structure Agent — pure deterministic price-action classifier.

Detects swing highs / swing lows via N-bar fractal rule, walks the last K pivots,
and classifies the trend into {strong_uptrend, uptrend, range, downtrend,
strong_downtrend} with a stage (early/middle/late/exhausted) and confidence.

Per the Phase 6 blinding boundary: this agent receives ONLY OHLCV bars + config.
It MUST NOT receive fundamentals, valuation, or any LLM-derived field.

No LLM. Pure pandas.
"""
from __future__ import annotations

from typing import Literal

import pandas as pd

from src.models.schemas import StructureAssessment, StructurePivot, StructureCfg


def _find_pivots(bars: pd.DataFrame, window: int) -> list[tuple[pd.Timestamp, float, str]]:
    """N-bar fractal: a swing high is a bar whose High exceeds the High of the N bars
    on each side; analogously for swing lows. Returns (timestamp, price, "high"|"low").
    """
    highs = bars["High"].values
    lows = bars["Low"].values
    out: list[tuple[pd.Timestamp, float, str]] = []
    for i in range(window, len(bars) - window):
        center_high = highs[i]
        center_low = lows[i]
        is_high = all(center_high > highs[j] for j in range(i - window, i)) and \
                  all(center_high > highs[j] for j in range(i + 1, i + window + 1))
        is_low = all(center_low < lows[j] for j in range(i - window, i)) and \
                 all(center_low < lows[j] for j in range(i + 1, i + window + 1))
        if is_high:
            out.append((bars.index[i], float(center_high), "high"))
        elif is_low:
            out.append((bars.index[i], float(center_low), "low"))
    return out


def _classify_pivots(pivots: list[tuple[pd.Timestamp, float, str]]) -> list[StructurePivot]:
    """Label each pivot as HH/HL/LH/LL by comparing to the previous pivot of the same kind."""
    last_high: float | None = None
    last_low: float | None = None
    classified: list[StructurePivot] = []
    for ts, price, kind in pivots:
        if kind == "high":
            label = "HH" if last_high is not None and price > last_high else "LH"
            last_high = price
        else:
            label = "HL" if last_low is not None and price > last_low else "LL"
            last_low = price
        classified.append(StructurePivot(date=ts.date().isoformat(), price=price, kind=label))
    return classified


def _trend_from_pivots(pivots: list[StructurePivot]) -> tuple[str, str, float]:
    """Walk the last K pivots and decide trend + stage + confidence."""
    if not pivots:
        return "range", "early", 0.0
    counts = {"HH": 0, "HL": 0, "LH": 0, "LL": 0}
    for p in pivots:
        counts[p.kind] += 1
    bullish = counts["HH"] + counts["HL"]
    bearish = counts["LH"] + counts["LL"]
    total = bullish + bearish
    if total == 0:
        return "range", "early", 0.0

    bull_ratio = bullish / total
    # Trend label
    if bull_ratio >= 0.85:
        trend = "strong_uptrend"
    elif bull_ratio >= 0.65:
        trend = "uptrend"
    elif bull_ratio >= 0.35:
        trend = "range"
    elif bull_ratio >= 0.15:
        trend = "downtrend"
    else:
        trend = "strong_downtrend"

    # Stage: look at the last 3 pivots. All same-polarity = late/exhausted; mixed = middle/early.
    recent = pivots[-3:]
    recent_bull = sum(1 for p in recent if p.kind in ("HH", "HL"))
    if trend in ("strong_uptrend", "uptrend"):
        stage = "late" if recent_bull >= len(recent) else "middle"
    elif trend in ("strong_downtrend", "downtrend"):
        stage = "late" if recent_bull == 0 else "middle"
    else:
        stage = "early"
    if total >= 6 and (bull_ratio >= 0.95 or bull_ratio <= 0.05):
        stage = "exhausted"

    # Confidence: distance from 50/50 split, scaled by sample size
    raw_conf = abs(bull_ratio - 0.5) * 2.0   # 0..1
    sample_weight = min(1.0, total / 6.0)    # need ~6+ pivots for full confidence
    confidence = raw_conf * sample_weight
    return trend, stage, round(confidence, 2)


def assess_structure(bars: pd.DataFrame, cfg: StructureCfg) -> StructureAssessment:
    """Run the full pipeline: detect pivots, classify, derive trend.

    `bars` must have columns High/Low/Close with a DatetimeIndex (the standard
    output of `fetch_market_bars`). The agent uses only High/Low.
    """
    if bars is None or bars.empty or len(bars) < (cfg.pivot_window_bars * 2 + 2):
        # Not enough data — return a low-confidence range verdict
        last_close = float(bars["Close"].iloc[-1]) if bars is not None and not bars.empty else 0.0
        return StructureAssessment(
            trend="range", stage="early", confidence=0.0,
            last_swing_high=last_close, last_swing_low=last_close,
            pivots=[],
            structure_summary="insufficient_history",
        )

    raw_pivots = _find_pivots(bars, cfg.pivot_window_bars)
    # Keep only the last K pivots for trend classification
    raw_pivots = raw_pivots[-cfg.pivots_to_evaluate:]
    classified = _classify_pivots(raw_pivots)

    trend, stage, confidence = _trend_from_pivots(classified)

    highs = [p.price for p in classified if p.kind in ("HH", "LH")]
    lows = [p.price for p in classified if p.kind in ("HL", "LL")]
    last_swing_high = float(max(highs)) if highs else float(bars["Close"].iloc[-1])
    last_swing_low = float(min(lows)) if lows else float(bars["Close"].iloc[-1])

    hh = sum(1 for p in classified if p.kind == "HH")
    hl = sum(1 for p in classified if p.kind == "HL")
    lh = sum(1 for p in classified if p.kind == "LH")
    ll = sum(1 for p in classified if p.kind == "LL")
    n_bars = len(bars)
    summary = (
        f"{n_bars} bars, {len(classified)} pivots evaluated "
        f"(HH={hh}, HL={hl}, LH={lh}, LL={ll}) — trend={trend} ({stage})"
    )

    return StructureAssessment(
        trend=trend, stage=stage, confidence=confidence,
        last_swing_high=last_swing_high, last_swing_low=last_swing_low,
        pivots=classified,
        structure_summary=summary,
    )
