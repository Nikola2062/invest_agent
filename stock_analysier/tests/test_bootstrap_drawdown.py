"""Tests for the non-LLM historical-bootstrap drawdown prior in risk_analyzer.

The bootstrap is the discipline layer's sanity floor: when the LLM's
P(drawdown ≥ X%) diverges from the empirical rolling-window base rate by more
than BOOTSTRAP_CLAMP_MAX_DIVERGENCE, we clamp toward the prior. Without it the
deterministic tactical ladder rides entirely on one LLM call's drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.risk_analyzer import (
    BOOTSTRAP_CLAMP_MAX_DIVERGENCE,
    bootstrap_drawdown_probabilities,
    clamp_against_bootstrap,
)


def _make_history(closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {"Close": closes},
        index=pd.date_range("2024-01-01", periods=len(closes), freq="D"),
    )


def test_bootstrap_returns_none_for_thin_history():
    """Less than horizon_days + BOOTSTRAP_MIN_WINDOWS → no prior."""
    history = _make_history([100.0] * 50)  # 50 bars, horizon 90 → insufficient
    out = bootstrap_drawdown_probabilities(history, horizon_days=90)
    assert out is None


def test_bootstrap_calm_market_low_drawdowns():
    """Slow uptrend with tiny pullbacks → all P(dd≥X%) should be low."""
    # 200 bars trending up 0.05%/day with no notable drawdowns
    closes = [100.0 * (1.005 ** i) for i in range(200)]
    history = _make_history(closes)
    out = bootstrap_drawdown_probabilities(history, horizon_days=30)
    assert out is not None
    # Monotonically uptrending series → no drawdowns
    assert out["10"] < 0.1
    assert out["15"] < 0.1
    assert out["20"] < 0.1
    assert out["25"] < 0.1


def test_bootstrap_crashy_market_high_drawdowns():
    """A series with a deep mid-window crash should produce elevated P(dd≥X%)."""
    # 100 flat, then a 40% drop over 30 bars, then flat 100 again.
    rng = np.random.default_rng(42)
    flat1 = [100.0 + rng.normal(0, 0.3) for _ in range(80)]
    crash = list(np.linspace(100, 60, 30))
    flat2 = [60.0 + rng.normal(0, 0.3) for _ in range(80)]
    closes = flat1 + crash + flat2
    history = _make_history(closes)
    out = bootstrap_drawdown_probabilities(history, horizon_days=30)
    assert out is not None
    # Many starting windows include some part of the crash
    assert out["10"] > 0.15, f"expected non-trivial P(dd≥10%), got {out['10']:.3f}"
    # Monotonicity by construction
    assert out["10"] >= out["15"] >= out["20"] >= out["25"]


def test_clamp_passes_through_within_tolerance():
    """When LLM and bootstrap agree, clamp returns the LLM value unchanged."""
    llm = {"10": 0.30, "15": 0.20, "20": 0.10, "25": 0.05}
    boot = {"10": 0.25, "15": 0.15, "20": 0.08, "25": 0.04}
    clamped, notes = clamp_against_bootstrap(llm, boot)
    assert clamped == llm
    assert notes == []


def test_clamp_pulls_overconfident_llm_toward_prior():
    """LLM says 90% chance of -25% drawdown but history says <5% → clamp it hard."""
    llm = {"10": 0.95, "15": 0.92, "20": 0.91, "25": 0.90}
    boot = {"10": 0.30, "15": 0.10, "20": 0.05, "25": 0.02}
    clamped, notes = clamp_against_bootstrap(llm, boot)
    # Every bucket should have moved
    assert len(notes) == 4
    # Clamped values stay within max_divergence of the bootstrap
    for k in ("10", "15", "20", "25"):
        assert abs(clamped[k] - boot[k]) <= BOOTSTRAP_CLAMP_MAX_DIVERGENCE + 1e-9
    # Direction: clamped should be lower than the LLM value
    for k in ("10", "15", "20", "25"):
        assert clamped[k] < llm[k]


def test_clamp_pushes_underconfident_llm_up():
    """LLM dismisses tail risk but history says crashes happen — clamp up."""
    llm = {"10": 0.01, "15": 0.01, "20": 0.01, "25": 0.01}
    # Each bootstrap value is > 0.20 above the LLM value → all 4 trigger
    boot = {"10": 0.60, "15": 0.50, "20": 0.40, "25": 0.30}
    clamped, notes = clamp_against_bootstrap(llm, boot)
    assert len(notes) == 4
    for k in ("10", "15", "20", "25"):
        assert clamped[k] > llm[k]


def test_clamp_respects_0_1_bounds():
    """Even with extreme bootstrap values, clamped output stays in [0, 1]."""
    llm = {"10": 0.99, "15": 0.99, "20": 0.99, "25": 0.99}
    boot = {"10": 0.99, "15": 0.99, "20": 0.99, "25": 0.99}
    clamped, _ = clamp_against_bootstrap(llm, boot)
    for v in clamped.values():
        assert 0.0 <= v <= 1.0


if __name__ == "__main__":
    test_bootstrap_returns_none_for_thin_history()
    test_bootstrap_calm_market_low_drawdowns()
    test_bootstrap_crashy_market_high_drawdowns()
    test_clamp_passes_through_within_tolerance()
    test_clamp_pulls_overconfident_llm_toward_prior()
    test_clamp_pushes_underconfident_llm_up()
    test_clamp_respects_0_1_bounds()
    print("✅ All bootstrap drawdown tests passed")
