"""Tests for the digest-suppression logic in notifier.scheduler._has_actionable_signal.

The daily digest cron is configured with send_only_if_action=true. The job should
SKIP the digest send on quiet days (no held tactical action, no watchlist BUY/WAIT,
no pending alerts) and FIRE on anything actionable.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from src.notifier import scheduler as sched
from src.models.schemas import (
    AnalysisResult,
    DevilAdvocateReview,
    FundamentalAssessment,
    HeldDecision,
    Holding,
    NotHeldDecision,
    Portfolio,
    RiskAssessment,
    Scenario,
    TacticalAction,
    ValuationResult,
)


def _result(symbol: str, market: str = "US",
            tactical_label: str | None = None,
            tactical_action: str = "no_action",
            not_held_rec: str = "PASS",
            da_verdict: str | None = None) -> AnalysisResult:
    ar = AnalysisResult(
        symbol=symbol, market=market,                       # type: ignore[arg-type]
        timestamp_utc=datetime.now(timezone.utc),
        current_price=100.0, currency="USD",
        position=None,
        fundamental=FundamentalAssessment(
            quality_score=7.0, moat_assessment="ok", moat_strength="narrow",
            balance_sheet_health="adequate", growth_outlook="ok",
            capital_allocation="ok", red_flags=[],
            thesis_one_liner="ok",
        ),
        valuation=ValuationResult(
            current_price=100.0, currency="USD",
            intrinsic_low=90, intrinsic_base=110, intrinsic_high=130,
            margin_of_safety_pct=10, methodology_notes="x", confidence="medium",
        ),
        risk=RiskAssessment(
            scenarios=[Scenario(name="base", probability=1.0,
                                expected_return_pct=5, expected_drawdown_pct=-5,
                                rationale="x")],
            drawdown_probabilities={"10": 0.1, "15": 0.05, "20": 0.02, "25": 0.01},
            realized_vol_annualized_pct=20.0,
            key_macro_signals=[], horizon_days=90,
        ),
        if_held=HeldDecision(
            tactical=TacticalAction(
                action=tactical_action,                     # type: ignore[arg-type]
                label=tactical_label, rationale="x",
            ),
        ),
        if_not_held=NotHeldDecision(
            recommendation=not_held_rec,                    # type: ignore[arg-type]
            rationale="x",
        ),
    )
    if da_verdict:
        ar.devil_advocate = DevilAdvocateReview(
            overall_verdict=da_verdict,                     # type: ignore[arg-type]
            summary="x", findings=[], counter_thesis="x",
        )
    return ar


def _patch_portfolio(monkeypatch: pytest.MonkeyPatch, held_symbols: list[tuple[str, str]]):
    """Make load_portfolio() inside the scheduler return a fake Portfolio."""
    portfolio = Portfolio(holdings=[
        Holding(symbol=s, market=m, shares=10, cost_basis_per_share=50.0,
                currency="USD" if m == "US" else "HKD")
        for s, m in held_symbols
    ])
    monkeypatch.setattr(sched, "load_portfolio", lambda: portfolio)


def test_suppress_when_everything_quiet(monkeypatch):
    _patch_portfolio(monkeypatch, [("AAA", "US"), ("BBB", "HK")])
    results = [
        _result("AAA"),                                     # held, no tactical
        _result("BBB", market="HK"),                        # held, no tactical
        _result("CCC", not_held_rec="PASS"),                # watch, PASS
    ]
    actionable, reason = sched._has_actionable_signal(results, None, alerts_pending=0)
    assert actionable is False
    assert "no held tactical action" in reason


def test_fire_on_pending_alerts(monkeypatch):
    _patch_portfolio(monkeypatch, [("AAA", "US")])
    results = [_result("AAA")]
    actionable, reason = sched._has_actionable_signal(results, None, alerts_pending=2)
    assert actionable is True
    assert "2 pending" in reason


def test_fire_on_held_tactical(monkeypatch):
    _patch_portfolio(monkeypatch, [("AAA", "US")])
    results = [_result("AAA", tactical_label="ORANGE_TRIM", tactical_action="trim")]
    actionable, reason = sched._has_actionable_signal(results, None, alerts_pending=0)
    assert actionable is True
    assert "AAA" in reason and "ORANGE_TRIM" in reason


def test_fire_on_watchlist_buy_now(monkeypatch):
    _patch_portfolio(monkeypatch, [])
    results = [_result("ZZZ", not_held_rec="BUY_NOW")]
    actionable, reason = sched._has_actionable_signal(results, None, alerts_pending=0)
    assert actionable is True
    assert "ZZZ" in reason and "BUY_NOW" in reason


def test_fire_on_devil_advocate_veto_on_held(monkeypatch):
    _patch_portfolio(monkeypatch, [("AAA", "US")])
    results = [_result("AAA", da_verdict="veto")]
    actionable, reason = sched._has_actionable_signal(results, None, alerts_pending=0)
    assert actionable is True
    assert "DA veto" in reason


def test_market_filter_excludes_other_markets(monkeypatch):
    """A US-filtered push should ignore HK-only activity."""
    _patch_portfolio(monkeypatch, [("BBB", "HK")])
    results = [_result("BBB", market="HK", tactical_label="ORANGE_TRIM", tactical_action="trim")]
    actionable_us, _ = sched._has_actionable_signal(results, "US", alerts_pending=0)
    assert actionable_us is False
    actionable_hk, _ = sched._has_actionable_signal(results, "HK", alerts_pending=0)
    assert actionable_hk is True


def test_market_filter_ALL_treated_as_no_filter(monkeypatch):
    _patch_portfolio(monkeypatch, [("BBB", "HK")])
    results = [_result("BBB", market="HK", tactical_label="ORANGE_TRIM", tactical_action="trim")]
    # The scheduler normalizes "ALL"→None before calling, but verify None and "ALL" both work
    for filt in (None, "ALL"):
        actionable, _ = sched._has_actionable_signal(results, filt, alerts_pending=0)
        assert actionable is True, f"filter={filt!r} should be treated as no filter"
