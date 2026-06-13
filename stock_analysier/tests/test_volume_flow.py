"""Tests for the Volume Agent — pure deterministic flow classifier."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.agents.volume_flow import assess_volume
from src.models.schemas import VolumeCfg


def _bars(closes: list[float], volumes: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({
        "Open": closes, "High": [c + 0.5 for c in closes],
        "Low": [c - 0.5 for c in closes], "Close": closes,
        "Volume": volumes,
    }, index=dates)


def test_accumulation_on_rising_obv_and_expansion():
    # 80 bars, prices drift up, volume expanding into the rally
    n = 80
    closes = [50 + i * 0.4 + 0.5 * np.sin(i / 3) for i in range(n)]
    # Heavier volume on up days
    volumes = []
    for i in range(n):
        diff = (closes[i] - closes[i - 1]) if i > 0 else 0.0
        base = 1_000_000 + i * 10_000          # ramp up
        bias = 600_000 if diff > 0 else 200_000
        volumes.append(base + bias)
    out = assess_volume(_bars(closes, volumes), VolumeCfg())
    assert out.institutional_flow == "accumulation", f"got {out.institutional_flow}; signals={out.signals}"
    assert out.obv_trend == "rising"
    print(f"✓ Up trend + expanding up-vol → accumulation (signals: {out.signals})")


def test_distribution_on_falling_obv_and_down_volume():
    n = 80
    closes = [80 - i * 0.4 + 0.5 * np.sin(i / 3) for i in range(n)]
    volumes = []
    for i in range(n):
        diff = (closes[i] - closes[i - 1]) if i > 0 else 0.0
        base = 1_000_000 + i * 10_000          # ramp up vol on the way down
        bias = 600_000 if diff < 0 else 200_000
        volumes.append(base + bias)
    out = assess_volume(_bars(closes, volumes), VolumeCfg())
    assert out.institutional_flow == "distribution", f"got {out.institutional_flow}; signals={out.signals}"
    assert out.obv_trend == "falling"
    print(f"✓ Down trend + expanding down-vol → distribution (signals: {out.signals})")


def test_neutral_on_flat_price_steady_volume():
    # Literally constant price → OBV is identically zero → no dominance.
    n = 80
    closes = [50.0] * n
    volumes = [1_000_000] * n
    out = assess_volume(_bars(closes, volumes), VolumeCfg())
    assert out.institutional_flow == "neutral", f"got {out.institutional_flow}; signals={out.signals}"
    assert out.obv_trend == "flat"
    print(f"✓ Constant price + steady vol → neutral")


def test_insufficient_history():
    out = assess_volume(_bars([50, 51, 52], [1_000_000] * 3), VolumeCfg())
    assert out.institutional_flow == "neutral"
    assert out.confidence == 0.0
    assert "insufficient_history" in out.signals
    print("✓ <20 bars → insufficient_history neutral")
