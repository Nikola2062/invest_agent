"""Unit tests for the on-demand deep-dive triage (``positions.triage_alarms``)
and the pure drawdown helper (``positions.drawdown_from_peak``).

Pure functions, no I/O — drawdowns/ranking are passed in as plain values,
mirroring the test style of test_position_overlay.py.
"""

from __future__ import annotations

import pytest

from tradingagents.portfolio.positions import (
    TriageConfig,
    drawdown_from_peak,
    triage_alarms,
)

pytestmark = pytest.mark.unit

CFG = TriageConfig()

# A two-name book; watchlist must never auto-trip.
BOOK = {
    "held": [
        {"symbol": "FIG", "cost_basis_per_share": 55.0, "shares": 2000},
        {"symbol": "0700.HK", "cost_basis_per_share": 500.0, "shares": 20000},
    ],
    "watchlist": {"US": [{"symbol": "NVDA"}]},
}

# Calm baseline: both names near their highs, healthy RS.
CALM_DD = {"FIG": -1.0, "0700.HK": -2.0}


def _syms(alarms):
    return {a["symbol"] for a in alarms}


# ----------------------------- drawdown_from_peak ----------------------------

def test_drawdown_at_fresh_high_is_zero():
    assert drawdown_from_peak([10, 20, 30]) == 0.0


def test_drawdown_off_high():
    # peak 100, last 88 -> -12%
    assert drawdown_from_peak([80, 100, 88]) == pytest.approx(-12.0)


def test_drawdown_ignores_nan_and_empty():
    assert drawdown_from_peak([float("nan"), 100, 90]) == pytest.approx(-10.0)
    assert drawdown_from_peak([]) is None
    assert drawdown_from_peak([float("nan")]) is None


# --------------------------------- calm day ----------------------------------

def test_calm_day_no_alarms():
    ranking = [{"symbol": "FIG", "rs_score": 3.0}, {"symbol": "0700.HK", "rs_score": 1.0}]
    assert triage_alarms(BOOK, CALM_DD, ranking, regime="Risk-On", macro_level="LOW") == []


# ------------------------------ drawdown alarm -------------------------------

def test_drawdown_below_threshold_trips():
    dd = {"FIG": -9.0, "0700.HK": -2.0}                  # FIG -9% < -8%
    alarms = triage_alarms(BOOK, dd, regime="Risk-On")
    assert _syms(alarms) == {"FIG"}
    assert "from high" in alarms[0]["reasons"][0]


def test_drawdown_just_above_threshold_stays_silent():
    dd = {"FIG": -6.4, "0700.HK": -2.0}                  # FIG -6.4% > -8%
    assert triage_alarms(BOOK, dd, regime="Risk-On") == []


def test_chronically_underwater_does_not_trip_without_fresh_drawdown():
    # Held name deeply below COST but sitting AT its recent high -> dd 0 -> silent.
    # This is the whole point of the change: cost basis is irrelevant to the alarm.
    dd = {"FIG": 0.0, "0700.HK": -1.0}
    assert triage_alarms(BOOK, dd, regime="Risk-On") == []


def test_missing_drawdown_cannot_trip():
    assert triage_alarms(BOOK, {}, regime="Risk-On") == []


# --------------------------------- RS alarm ----------------------------------

def test_weak_rs_trips():
    ranking = [{"symbol": "FIG", "rs_score": 2.0}, {"symbol": "0700.HK", "rs_score": -7.0}]
    alarms = triage_alarms(BOOK, CALM_DD, ranking, regime="Risk-On")
    assert _syms(alarms) == {"0700.HK"}
    assert "RS" in alarms[0]["reasons"][0]


def test_none_rs_score_does_not_trip():
    ranking = [{"symbol": "FIG", "rs_score": None}]
    assert triage_alarms(BOOK, CALM_DD, ranking, regime="Risk-On") == []


# ------------------------------- danger gate ---------------------------------

def test_risk_off_regime_flags_all_holdings():
    alarms = triage_alarms(BOOK, CALM_DD, regime="Risk-Off Defensive")
    assert _syms(alarms) == {"FIG", "0700.HK"}
    assert all("risk-off" in a["reasons"] for a in alarms)


def test_danger_macro_level_flags_all_holdings():
    alarms = triage_alarms(BOOK, CALM_DD, regime="Risk-On", macro_level="CRITICAL")
    assert _syms(alarms) == {"FIG", "0700.HK"}


def test_macro_level_case_insensitive():
    assert _syms(triage_alarms(BOOK, CALM_DD, macro_level="high")) == {"FIG", "0700.HK"}


# ------------------------ watchlist & multi-reason ---------------------------

def test_watchlist_names_never_auto_trip():
    alarms = triage_alarms(BOOK, CALM_DD, regime="Risk-Off Defensive")
    assert "NVDA" not in _syms(alarms)


def test_multiple_reasons_accumulate():
    dd = {"FIG": -12.0, "0700.HK": -1.0}                 # FIG off-high + weak RS + risk-off
    ranking = [{"symbol": "FIG", "rs_score": -9.0}]
    alarms = triage_alarms(BOOK, dd, ranking, regime="Risk-Off Defensive")
    fig = next(a for a in alarms if a["symbol"] == "FIG")
    assert len(fig["reasons"]) == 3


# --------------------------------- config ------------------------------------

def test_custom_thresholds_respected():
    dd = {"FIG": -3.6, "0700.HK": -1.0}                  # only trips a looser threshold
    loose = TriageConfig(drawdown_alarm_pct=-3.0)
    assert _syms(triage_alarms(BOOK, dd, regime="Risk-On", cfg=loose)) == {"FIG"}


def test_empty_book_is_empty():
    assert triage_alarms({"held": []}, {"FIG": -50.0}, regime="Risk-Off Defensive") == []
