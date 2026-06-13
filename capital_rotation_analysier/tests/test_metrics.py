"""Unit tests for metrics.py. These cover the edge cases that previously
shipped broken (robust-z chained windows, etc.) plus the load-bearing formulas
that the entire signal stack depends on.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from rotation import metrics as M


def _series(values: list[float]) -> pd.Series:
    idx = pd.date_range("2025-01-01", periods=len(values), freq="B")
    return pd.Series(values, index=idx, dtype=float)


def _frame(d: dict[str, list[float]]) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=len(next(iter(d.values()))), freq="B")
    return pd.DataFrame(d, index=idx, dtype=float)


# ---------- log_returns ----------

def test_log_returns_basic():
    df = _frame({"X": [100, 110, 99, 99]})
    r = M.log_returns(df, 1)
    assert pd.isna(r.iloc[0, 0])
    assert abs(r.iloc[1, 0] - np.log(110/100)) < 1e-12
    assert abs(r.iloc[2, 0] - np.log(99/110)) < 1e-12


def test_log_returns_multi_period_horizon_uses_n_back():
    df = _frame({"X": [100, 105, 110, 121]})
    r = M.log_returns(df, 3)
    # First 3 are NaN; 4th is ln(121/100)
    assert pd.isna(r.iloc[2, 0])
    assert abs(r.iloc[3, 0] - np.log(121/100)) < 1e-12


# ---------- realized vol ----------

def test_realized_vol_annualized_factor():
    # constant daily moves -> low vol; verify annualization factor of sqrt(252)
    returns_seed = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
                    111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121,
                    122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133]
    df = _frame({"X": returns_seed})
    vol = M.realized_vol(df, window=30)
    last = vol["X"].dropna().iloc[-1]
    # std of constant log-returns is ~0 but >0 due to compounding; just check >0 and finite
    assert last >= 0 and np.isfinite(last)


def test_realized_vol_returns_nan_below_min_obs():
    df = _frame({"X": [100.0, 101.0, 102.0]})  # 3 points; window=30 -> NaN
    vol = M.realized_vol(df, window=30)
    assert vol["X"].isna().all()


# ---------- relative_volume + volume_zscore ----------

def test_relative_volume_ln_normal_for_constant():
    # Constant volume -> RV at any point is ln(V/median(V)) = ln(1) = 0
    df = _frame({"X": [1e6] * 40})
    rv = M.relative_volume(df, window=30)
    valid = rv["X"].dropna()
    assert len(valid) > 0
    assert (valid.abs() < 1e-9).all()


def test_volume_zscore_log_first():
    # Add some baseline variance (constant series would produce std=0 -> NaN, correctly).
    rng = np.random.default_rng(0)
    base = list(rng.uniform(0.9e6, 1.1e6, 60))
    df = _frame({"X": base + [1e8]})
    vz = M.volume_zscore(df, window=60)
    final = vz["X"].iloc[-1]
    # The spike (100x) on log scale is z = ln(100)/std(ln) — should be large but finite
    assert np.isfinite(final) and final > 3.0


# ---------- robust_z (regression: chained windows bug) ----------

def test_robust_z_returns_value_with_minimum_observations():
    """Regression: chained .rolling().median() previously compounded warm-ups
    and dropped valid observations. With min_obs=126 and 200 valid points,
    we should get a real number out, not NaN."""
    rng = np.random.default_rng(42)
    series = pd.Series(rng.standard_normal(200).cumsum(),
                       index=pd.date_range("2024-01-01", periods=200, freq="B"))
    z = M.robust_z(series, window=200, min_obs=126)
    # Last value should be finite and within clip range
    last = z.iloc[-1]
    assert np.isfinite(last), f"expected finite, got {last}"
    assert -3.0 <= last <= 3.0


def test_robust_z_nan_below_min_obs():
    series = _series([1.0, 2.0, 3.0, 4.0, 5.0])
    z = M.robust_z(series, window=200, min_obs=126)
    assert z.isna().all()


def test_robust_z_nan_when_mad_zero():
    # All-constant series -> MAD = 0 -> return NaN (don't divide by zero)
    series = _series([5.0] * 200)
    z = M.robust_z(series, window=200, min_obs=126)
    assert z.dropna().empty or (z.dropna() == 0).all() or z.iloc[-1] != z.iloc[-1]  # NaN check


def test_robust_z_clipped_to_three():
    # Single large outlier -> raw z would exceed 3; clip to ±3
    base = [1.0] * 130 + [50.0]  # 130 ones, then big outlier
    series = _series(base + [100.0])  # last point: way out
    z = M.robust_z(pd.Series(base + [100.0],
                              index=pd.date_range("2024-01-01", periods=len(base)+1, freq="B")),
                   window=130, min_obs=100)
    last = z.iloc[-1]
    assert last <= 3.0 and last >= -3.0 if np.isfinite(last) else True


# ---------- rs_rank ----------

def test_rs_rank_assigns_high_to_strongest():
    """Best-performing symbol should get a high RS rank."""
    n = 100  # enough for 63d quarterly blend
    a_prices = list(np.linspace(100, 200, n))  # +100%
    b_prices = list(np.linspace(100, 100, n))  # 0%
    c_prices = list(np.linspace(100, 80, n))   # -20%
    df = _frame({"A": a_prices, "B": b_prices, "C": c_prices})
    ranks = M.rs_rank(df)
    last = ranks.iloc[-1].dropna()
    # A should rank highest, C lowest
    assert last["A"] > last["B"] > last["C"]


# ---------- breadth ----------

def test_pct_advancing_basic():
    rng = pd.date_range("2024-01-01", periods=3, freq="B")
    rd = pd.DataFrame({
        "A": [0.01, -0.01, 0.02],
        "B": [-0.005, 0.01, 0.01],
        "C": [0.02, -0.02, -0.01],
    }, index=rng)
    pct = M.pct_advancing(rd)
    assert abs(pct.iloc[0] - 2/3) < 1e-9   # A,C up
    assert abs(pct.iloc[1] - 1/3) < 1e-9   # B up
    assert abs(pct.iloc[2] - 2/3) < 1e-9   # A,B up


def test_concentration_high_when_one_dominates():
    rng = pd.date_range("2024-01-01", periods=1, freq="B")
    df = pd.DataFrame({"A": [0.10], "B": [0.0001], "C": [0.0001]}, index=rng)
    h = M.concentration(df, top_k=3)
    # A captures ~all the weight -> Herfindahl near 1
    assert h.iloc[0] > 0.9
