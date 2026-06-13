"""Volume Agent — pure deterministic accumulation / distribution classifier.

Signals:
  - OBV trend over the last N days (linear-regression slope, normalized)
  - 20-day vs 50-day average volume (expansion / contraction)
  - Up-day-volume vs down-day-volume ratio (UDV / DDV)
  - Most-recent volume spike multiple (vs trailing 60d average)

Output: `institutional_flow` ∈ {accumulation, distribution, neutral}.

No LLM. Pure pandas / numpy. Receives only OHLCV bars + config (blinded from
the fundamental side per the Phase 6 architecture).
"""
from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

from src.models.schemas import VolumeAssessment, VolumeCfg


def _obv(closes: pd.Series, volumes: pd.Series) -> pd.Series:
    """Standard On-Balance Volume: cumulative sum of signed volume."""
    direction = np.sign(closes.diff().fillna(0.0))
    return (direction * volumes).cumsum()


def _slope_normalized(series: pd.Series) -> float:
    """Linear regression slope of `series` vs t, divided by mean(|series|) to make
    the result scale-invariant. Positive = rising, negative = falling.
    """
    if len(series) < 5:
        return 0.0
    y = series.values.astype(float)
    x = np.arange(len(y), dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    denom = float(np.mean(np.abs(y))) or 1.0
    return slope / denom


def _up_down_volume_ratio(closes: pd.Series, volumes: pd.Series, lookback: int) -> float:
    """Sum of volume on up-days divided by sum on down-days, over the last `lookback` bars."""
    diffs = closes.diff().tail(lookback)
    vols = volumes.tail(lookback)
    up = float(vols[diffs > 0].sum())
    down = float(vols[diffs < 0].sum())
    if down <= 0:
        return float("inf") if up > 0 else 1.0
    return up / down


def assess_volume(bars: pd.DataFrame, cfg: VolumeCfg) -> VolumeAssessment:
    if bars is None or bars.empty or "Volume" not in bars.columns or len(bars) < 20:
        return VolumeAssessment(
            institutional_flow="neutral",
            obv_trend="flat",
            volume_expansion_pct=0.0,
            up_down_volume_ratio=1.0,
            last_earnings_volume_spike_x=None,
            confidence=0.0,
            signals=["insufficient_history"],
        )

    closes = bars["Close"].astype(float)
    volumes = bars["Volume"].astype(float).fillna(0.0)

    # 1. OBV trend
    obv_series = _obv(closes, volumes).tail(cfg.obv_lookback_days)
    obv_slope_norm = _slope_normalized(obv_series)
    if obv_slope_norm > 0.0015:
        obv_trend = "rising"
    elif obv_slope_norm < -0.0015:
        obv_trend = "falling"
    else:
        obv_trend = "flat"

    # 2. 20d vs 50d volume expansion
    short_avg = float(volumes.tail(cfg.expansion_window_short).mean())
    long_avg = float(volumes.tail(cfg.expansion_window_long).mean())
    expansion_pct = ((short_avg / long_avg) - 1.0) * 100.0 if long_avg > 0 else 0.0

    # 3. Up/Down volume ratio over the OBV lookback window
    udv_ddv = _up_down_volume_ratio(closes, volumes, lookback=cfg.obv_lookback_days)

    # 4. Spike: max volume in last 5 bars vs trailing 60d average
    spike_window = volumes.tail(5)
    baseline = float(volumes.tail(60).head(55).mean()) or 1.0
    spike_x: float | None = float(spike_window.max() / baseline) if baseline > 0 else None
    if spike_x is not None and spike_x < 1.5:
        spike_x = None   # only report meaningful spikes

    # Composite classification
    signals: list[str] = []
    bullish_score = 0
    bearish_score = 0

    if obv_trend == "rising":
        bullish_score += 2; signals.append("OBV rising")
    elif obv_trend == "falling":
        bearish_score += 2; signals.append("OBV falling")

    if expansion_pct > 15:
        bullish_score += 1; signals.append(f"Volume expansion +{expansion_pct:.0f}%")
    elif expansion_pct < -15:
        bearish_score += 1; signals.append(f"Volume contraction {expansion_pct:.0f}%")

    if udv_ddv == float("inf") or udv_ddv > 1.5:
        bullish_score += 1
        signals.append(f"UDV/DDV {udv_ddv if udv_ddv != float('inf') else 'inf'}")
    elif udv_ddv < 0.67:
        bearish_score += 1
        signals.append(f"UDV/DDV {udv_ddv:.2f}")

    if spike_x is not None:
        signals.append(f"Recent volume spike {spike_x:.1f}× baseline")

    if bullish_score - bearish_score >= 2:
        institutional_flow: Literal["accumulation", "distribution", "neutral"] = "accumulation"
    elif bearish_score - bullish_score >= 2:
        institutional_flow = "distribution"
    else:
        institutional_flow = "neutral"

    # Confidence = magnitude of dominance, scaled by sample size
    score_diff = abs(bullish_score - bearish_score)
    confidence = min(1.0, score_diff / 4.0) * (1.0 if len(bars) >= 60 else len(bars) / 60.0)
    confidence = round(confidence, 2)

    return VolumeAssessment(
        institutional_flow=institutional_flow,
        obv_trend=obv_trend,
        volume_expansion_pct=round(expansion_pct, 1),
        up_down_volume_ratio=round(udv_ddv, 2) if udv_ddv != float("inf") else 999.0,
        last_earnings_volume_spike_x=round(spike_x, 1) if spike_x else None,
        confidence=confidence,
        signals=signals,
    )
