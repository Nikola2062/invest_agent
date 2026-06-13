"""Tests for the IC validation harness — the math that decides whether a
signal gets published or suppressed."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from rotation import validate as V


def _series(values, start="2024-01-01"):
    idx = pd.date_range(start, periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


def test_rolling_spearman_ic_perfect_correlation():
    """Perfectly aligned score and forward return -> rolling IC ~= 1.0."""
    rng = np.random.default_rng(0)
    score = _series(rng.standard_normal(200))
    fwd = score * 0.5  # monotonic transform -> rank correlation = 1
    ic = V.rolling_spearman_ic(score, fwd, window=63)
    last = ic.dropna().iloc[-1]
    assert last > 0.99, f"expected ~1.0 IC for monotonic input, got {last}"


def test_rolling_spearman_ic_zero_for_independent():
    rng = np.random.default_rng(1)
    score = _series(rng.standard_normal(200))
    fwd = _series(rng.standard_normal(200))
    ic = V.rolling_spearman_ic(score, fwd, window=63)
    median_ic = ic.median()
    # Independent series -> IC near 0
    assert abs(median_ic) < 0.20, f"independent series should have small IC, got {median_ic}"


def test_assess_directional_returns_undetermined_below_min_obs():
    score = _series([1.0, 2.0, 3.0])
    close = pd.DataFrame({"SPY": [100.0, 101.0, 102.0]},
                         index=pd.date_range("2024-01-01", periods=3, freq="B"))
    vol_proxy = _series([0.1, 0.1, 0.1])
    v = V.assess_directional("test_sig", date.today(), score, close, "SPY", vol_proxy)
    assert v.verdict == "undetermined"
    assert "insufficient_history" in v.reason


def test_hit_rate_buckets_handles_empty():
    out = V.hit_rate_buckets(pd.Series(dtype=float), pd.Series(dtype=float), pd.Series(dtype=float))
    assert out["n"] == 0
    assert pd.isna(out["overall"]) or out["overall"] != out["overall"]  # NaN


def test_hit_rate_buckets_full_agreement():
    """When score and forward return always agree on sign, hit rate = 1.0."""
    n = 30
    rng = np.random.default_rng(7)
    rand = rng.standard_normal(n)
    score = _series(rand)
    fwd = _series(rand * 2)  # same sign always
    vol = _series([0.1 + i * 0.001 for i in range(n)])
    out = V.hit_rate_buckets(score, fwd, vol)
    assert abs(out["overall"] - 1.0) < 1e-9
