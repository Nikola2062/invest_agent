"""Tests for the verdict-override logic in src.agents.devil_advocate.

The LLM returns a DevilAdvocateReview; the agent then DETERMINISTICALLY rewrites
overall_verdict + veto_reason based on the finding mix:
  - any veto-finding   -> veto
  - >=3 concern        -> veto (independent concerns are themselves a fatal pattern)
  - 1-2 concern        -> pass_with_concerns
  - only info / none   -> pass

We mock chat_json so this runs without an LLM key and tests only the override.
"""
from __future__ import annotations

from datetime import date, datetime, timezone
from typing import Callable

import pytest

from src.agents import devil_advocate
from src.models.schemas import (
    AnalysisResult,
    DevilAdvocateReview,
    DevilFinding,
    FundamentalAssessment,
    HeldDecision,
    Holding,
    NotHeldDecision,
    OrderSpec,
    RiskAssessment,
    Scenario,
    TacticalAction,
    ValuationResult,
)


# ---------- helpers ----------

def _make_result() -> AnalysisResult:
    """Minimal AnalysisResult — Devil's Advocate only reads, never depends on values."""
    return AnalysisResult(
        symbol="TEST", market="US",
        timestamp_utc=datetime.now(timezone.utc),
        current_price=100.0, currency="USD",
        position=Holding(symbol="TEST", market="US", shares=10,
                         cost_basis_per_share=80.0, currency="USD"),
        fundamental=FundamentalAssessment(
            quality_score=8.0, moat_assessment="strong network effects",
            moat_strength="wide", balance_sheet_health="strong",
            growth_outlook="durable", capital_allocation="disciplined",
            red_flags=[], thesis_one_liner="High-quality compounder.",
        ),
        valuation=ValuationResult(
            current_price=100.0, currency="USD",
            intrinsic_low=110.0, intrinsic_base=130.0, intrinsic_high=150.0,
            margin_of_safety_pct=30.0, methodology_notes="DCF + multiples",
            confidence="medium",
        ),
        risk=RiskAssessment(
            scenarios=[Scenario(name="base", probability=1.0,
                                expected_return_pct=10, expected_drawdown_pct=-10,
                                rationale="base")],
            drawdown_probabilities={"10": 0.3, "15": 0.2, "20": 0.1, "25": 0.05},
            realized_vol_annualized_pct=25.0,
            key_macro_signals=[], horizon_days=90,
        ),
        if_held=HeldDecision(
            tactical=TacticalAction(action="no_action", rationale="hold"),
        ),
        if_not_held=NotHeldDecision(recommendation="WAIT_FOR_PRICE", rationale="wait"),
    )


def _finding(severity: str, category: str = "data_quality", finding: str = "x") -> DevilFinding:
    return DevilFinding(
        category=category,                       # type: ignore[arg-type]
        severity=severity,                       # type: ignore[arg-type]
        finding=finding,
        evidence="evidence text",
        recommendation="recommendation text",
    )


def _stub_chat_json(monkeypatch: pytest.MonkeyPatch, returned: DevilAdvocateReview) -> dict:
    """Replace chat_json in the agent module so no LLM call is made.

    Returns a dict that captures call kwargs for assertion (system / user / schema / temperature / reasoner).
    """
    captured: dict = {}

    def fake(**kwargs):
        captured.update(kwargs)
        # chat_json normally re-validates; here we hand back the exact object the test wants.
        return returned

    monkeypatch.setattr(devil_advocate, "chat_json", fake)
    return captured


# ---------- tests ----------

def test_single_veto_finding_forces_veto(monkeypatch):
    """Any finding with severity=veto -> overall_verdict=veto regardless of LLM-claimed verdict."""
    llm_says = DevilAdvocateReview(
        overall_verdict="pass",                         # LLM tries to soft-pedal
        summary="looks fine",
        findings=[_finding("veto", "moat_erosion", "Moat is broken.")],
        counter_thesis="Bears would point out the moat erosion.",
        veto_reason=None,
    )
    _stub_chat_json(monkeypatch, llm_says)

    out = devil_advocate.review(_make_result())

    assert out.overall_verdict == "veto"
    assert out.veto_reason is not None and "Moat is broken" in out.veto_reason


def test_three_concerns_forces_veto(monkeypatch):
    """Three or more concern-level findings on independent failure modes -> veto."""
    llm_says = DevilAdvocateReview(
        overall_verdict="pass_with_concerns",            # LLM only said concerns
        summary="some issues but ok",
        findings=[
            _finding("concern", "valuation_optimism", "Bull case priced in."),
            _finding("concern", "macro_blind_spot", "Rate-cycle risk."),
            _finding("concern", "data_quality", "Thin segment disclosure."),
        ],
        counter_thesis="The bear case is the concerns above stacked up.",
        veto_reason=None,
    )
    _stub_chat_json(monkeypatch, llm_says)

    out = devil_advocate.review(_make_result())

    assert out.overall_verdict == "veto"
    assert out.veto_reason is not None
    # Reason references the categories that piled up
    assert any(cat in out.veto_reason for cat in
               ("valuation_optimism", "macro_blind_spot", "data_quality"))


def test_two_concerns_is_pass_with_concerns(monkeypatch):
    """Below the 3-concern threshold and no veto-finding -> pass_with_concerns."""
    llm_says = DevilAdvocateReview(
        overall_verdict="pass",
        summary="mostly clean",
        findings=[
            _finding("concern", "valuation_optimism", "Optimistic terminal multiple."),
            _finding("concern", "data_quality", "Sparse segment reporting."),
        ],
        counter_thesis="Bear: valuation optimism is real.",
        veto_reason=None,
    )
    _stub_chat_json(monkeypatch, llm_says)

    out = devil_advocate.review(_make_result())

    assert out.overall_verdict == "pass_with_concerns"


def test_only_info_findings_is_pass(monkeypatch):
    """Info-only findings should leave overall_verdict at pass."""
    llm_says = DevilAdvocateReview(
        overall_verdict="veto",                         # LLM is wrong; override down
        summary="nothing serious",
        findings=[
            _finding("info", "narrative_fallacy", "Story is clean — note for awareness."),
        ],
        counter_thesis="Bear case is weak.",
        veto_reason="LLM tried to veto",
    )
    _stub_chat_json(monkeypatch, llm_says)

    out = devil_advocate.review(_make_result())

    assert out.overall_verdict == "pass"


def test_no_findings_is_pass(monkeypatch):
    """Empty findings list -> pass."""
    llm_says = DevilAdvocateReview(
        overall_verdict="pass_with_concerns",
        summary="actually clean",
        findings=[],
        counter_thesis="Bears would struggle.",
        veto_reason=None,
    )
    _stub_chat_json(monkeypatch, llm_says)

    out = devil_advocate.review(_make_result())

    assert out.overall_verdict == "pass"


def test_veto_reason_preserved_if_llm_provided(monkeypatch):
    """If the LLM already supplied a veto_reason on a veto-finding, override should keep it."""
    llm_says = DevilAdvocateReview(
        overall_verdict="veto",
        summary="real problem",
        findings=[_finding("veto", "circle_of_competence", "Out of circle.")],
        counter_thesis="The bear case is straightforward.",
        veto_reason="Pre-existing reason from LLM",
    )
    _stub_chat_json(monkeypatch, llm_says)

    out = devil_advocate.review(_make_result())

    assert out.overall_verdict == "veto"
    assert out.veto_reason == "Pre-existing reason from LLM"


def test_mix_of_veto_and_concern_still_veto(monkeypatch):
    """A single veto plus concerns is still veto; veto wins regardless of concern count."""
    llm_says = DevilAdvocateReview(
        overall_verdict="pass_with_concerns",
        summary="mixed",
        findings=[
            _finding("veto", "valuation_optimism", "Terminal multiple absurd."),
            _finding("concern", "macro_blind_spot", "Rate path uncertain."),
        ],
        counter_thesis="Bear: the terminal multiple alone breaks the thesis.",
        veto_reason=None,
    )
    _stub_chat_json(monkeypatch, llm_says)

    out = devil_advocate.review(_make_result())

    assert out.overall_verdict == "veto"
    assert out.veto_reason is not None and "Terminal multiple" in out.veto_reason


def _result_with_orders() -> AnalysisResult:
    """AnalysisResult carrying real orders on both held / not-held paths."""
    ar = _make_result()
    ar.if_held = HeldDecision(
        tactical=TacticalAction(
            level=2, label="ORANGE_TRIM", action="trim",
            trim_pct_of_position=30.0,
            rebuy_band_low=51.0, rebuy_band_high=54.0,
            hedge_recommended=False,
            rationale="Original tactical rationale here.",
        ),
        immediate_orders=[
            OrderSpec(side="SELL", quantity=3, symbol="TEST",
                      order_type="LIMIT", limit_price=99.5,
                      rationale="trim 30%"),
        ],
        rebuy_orders=[
            OrderSpec(side="BUY", quantity=2, symbol="TEST",
                      order_type="LIMIT", limit_price=54.0,
                      conditional=True, rationale="tranche 1"),
            OrderSpec(side="BUY", quantity=1, symbol="TEST",
                      order_type="LIMIT", limit_price=51.0,
                      conditional=True, rationale="tranche 2"),
        ],
    )
    ar.if_not_held = NotHeldDecision(
        recommendation="BUY_NOW",
        entry_orders=[
            OrderSpec(side="BUY", quantity=100, symbol="TEST",
                      order_type="LIMIT", limit_price=100.5,
                      rationale="MoS 30%"),
        ],
        rationale="Original entry rationale.",
    )
    return ar


def test_apply_veto_clears_held_orders_and_rationale():
    """Veto verdict must blank immediate + rebuy orders and prepend [DA VETO] marker."""
    ar = _result_with_orders()
    dr = DevilAdvocateReview(
        overall_verdict="veto",
        summary="Material flaw.",
        findings=[_finding("veto", "moat_erosion", "Moat broken.")],
        counter_thesis="bear case",
        veto_reason="Moat is broken; revenue concentration risk.",
    )

    applied = devil_advocate.apply_veto(ar, dr)

    assert applied is True
    assert ar.if_held.immediate_orders == []
    assert ar.if_held.rebuy_orders == []
    assert "DEVIL'S ADVOCATE VETO" in ar.if_held.tactical.rationale
    assert "Moat is broken" in ar.if_held.tactical.rationale
    # Keep the label so the dashboard can still show "would have been ORANGE_TRIM"
    assert ar.if_held.tactical.label == "ORANGE_TRIM"


def test_apply_veto_forces_not_held_to_pass():
    """Veto on a BUY_NOW recommendation must downgrade to PASS with cleared entries."""
    ar = _result_with_orders()
    dr = DevilAdvocateReview(
        overall_verdict="veto",
        summary="Bad buy.",
        findings=[_finding("veto", "valuation_optimism", "Terminal multiple absurd.")],
        counter_thesis="bear",
        veto_reason="Valuation is fantasy.",
    )

    devil_advocate.apply_veto(ar, dr)

    assert ar.if_not_held.recommendation == "PASS"
    assert ar.if_not_held.entry_orders == []
    assert "DEVIL'S ADVOCATE VETO" in ar.if_not_held.rationale
    assert "was BUY_NOW" in ar.if_not_held.rationale


def test_apply_veto_no_op_when_not_veto():
    """pass / pass_with_concerns must NOT touch orders."""
    ar = _result_with_orders()
    for verdict in ("pass", "pass_with_concerns"):
        dr = DevilAdvocateReview(
            overall_verdict=verdict,  # type: ignore[arg-type]
            summary="ok", findings=[], counter_thesis="x", veto_reason=None,
        )
        applied = devil_advocate.apply_veto(ar, dr)
        assert applied is False
        assert len(ar.if_held.immediate_orders) == 1
        assert ar.if_not_held.recommendation == "BUY_NOW"


def test_apply_veto_is_idempotent():
    """Applying veto twice must not double-prefix the rationale."""
    ar = _result_with_orders()
    dr = DevilAdvocateReview(
        overall_verdict="veto", summary="x",
        findings=[_finding("veto", "data_quality", "thin data")],
        counter_thesis="x", veto_reason="thin data",
    )

    devil_advocate.apply_veto(ar, dr)
    rationale_after_first = ar.if_held.tactical.rationale
    devil_advocate.apply_veto(ar, dr)

    assert ar.if_held.tactical.rationale == rationale_after_first
    assert ar.if_held.tactical.rationale.count("DEVIL'S ADVOCATE VETO") == 1


def test_review_passes_reasoner_and_temperature_to_llm(monkeypatch):
    """Sanity check: the agent calls chat_json with reasoner=True and the expected temperature."""
    llm_says = DevilAdvocateReview(
        overall_verdict="pass", summary="s",
        findings=[], counter_thesis="c", veto_reason=None,
    )
    captured = _stub_chat_json(monkeypatch, llm_says)

    devil_advocate.review(_make_result(), use_reasoner=True)

    assert captured.get("reasoner") is True
    assert captured.get("temperature") == 0.3
    assert captured.get("schema") is DevilAdvocateReview
