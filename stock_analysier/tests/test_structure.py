"""Tests for the Structure Agent — pure deterministic pivot detection."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.agents.structure import assess_structure
from src.models.schemas import StructureCfg


def _bars_from_closes(closes: list[float]) -> pd.DataFrame:
    """Build a synthetic OHLC DataFrame from a close series. High = close + 0.5,
    Low = close - 0.5 — enough wiggle room for the N-bar fractal to find pivots.
    """
    dates = pd.date_range("2025-01-01", periods=len(closes), freq="D")
    return pd.DataFrame({
        "Open": closes, "High": [c + 0.5 for c in closes],
        "Low": [c - 0.5 for c in closes], "Close": closes,
        "Volume": [1_000_000] * len(closes),
    }, index=dates)


def _zigzag(start: float, swings: list[float], dwell: int = 6) -> list[float]:
    """Generate a series that walks from `start` through each swing level,
    linearly interpolating over `dwell` bars between turning points.
    """
    out: list[float] = [start]
    cur = start
    for tgt in swings:
        step = (tgt - cur) / dwell
        for _ in range(dwell):
            cur = cur + step
            out.append(cur)
    return out


def test_strong_uptrend_HH_HL_pattern():
    # zig-zag UP: each new swing high > previous, each new low > previous
    closes = _zigzag(50, [60, 55, 70, 65, 80, 75, 90])
    bars = _bars_from_closes(closes)
    out = assess_structure(bars, StructureCfg(pivot_window_bars=2, pivots_to_evaluate=8))
    assert out.trend in ("strong_uptrend", "uptrend"), f"got {out.trend}; pivots={[p.kind for p in out.pivots]}"
    assert any(p.kind == "HH" for p in out.pivots), [p.kind for p in out.pivots]
    print(f"✓ Uptrend zigzag → {out.trend} (confidence {out.confidence})")


def test_strong_downtrend_LH_LL_pattern():
    closes = _zigzag(100, [80, 90, 70, 80, 60, 70, 50])
    bars = _bars_from_closes(closes)
    out = assess_structure(bars, StructureCfg(pivot_window_bars=2, pivots_to_evaluate=8))
    assert out.trend in ("strong_downtrend", "downtrend"), f"got {out.trend}; pivots={[p.kind for p in out.pivots]}"
    assert any(p.kind in ("LL", "LH") for p in out.pivots)
    print(f"✓ Downtrend zigzag → {out.trend} (confidence {out.confidence})")


def test_range_pattern_returns_range():
    # oscillate around a mean, no directional drift
    closes = _zigzag(50, [55, 48, 53, 47, 54, 49, 53])
    bars = _bars_from_closes(closes)
    out = assess_structure(bars, StructureCfg(pivot_window_bars=2, pivots_to_evaluate=8))
    # Any of range / weak uptrend / weak downtrend is acceptable, but confidence should be low
    assert out.confidence < 0.7, f"expected low confidence on range, got {out.confidence}"
    print(f"✓ Range pattern → trend={out.trend} (confidence {out.confidence:.2f})")


def test_insufficient_history():
    bars = _bars_from_closes([50, 51, 52])
    out = assess_structure(bars, StructureCfg(pivot_window_bars=5))
    assert out.trend == "range"
    assert out.confidence == 0.0
    assert out.structure_summary == "insufficient_history"
    print("✓ Insufficient history → range / 0.0 confidence")
