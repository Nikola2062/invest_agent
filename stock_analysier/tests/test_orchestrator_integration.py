"""End-to-end orchestrator integration test.

Patches all upstream data-fetchers and `chat_json` so analyze() runs in-process
without network or LLM calls. Verifies wiring contracts that unit tests
individually don't catch:

  * Pipeline completes, produces a valid AnalysisResult with expected fields.
  * Devil's Advocate veto actually CLEARS orders (the architecture diagram claim).
  * Bootstrap drawdown prior runs and surfaces on the result.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.catalysts import CatalystSnapshot
from src.data.fundamentals import FundamentalsSnapshot
from src.data.macro import MacroSnapshot
from src.data.prices import PriceSnapshot
from src.data.sentiment import SentimentSnapshot
from src.models.schemas import (
    ContrarianAssessment,
    DevilAdvocateReview,
    DevilFinding,
    FinancialReport,
    ForwardCatalysts,
    ForwardScenarios,
    FundamentalAssessment,
    HedgePlan,
    PriceMap,
    PriceMapZone,
    PriceScenario,
    RiskAssessment,
    Scenario,
    ValuationResult,
)


# ---------- Stub data ----------

def _stub_history(n_days: int = 260, start_price: float = 100.0) -> pd.DataFrame:
    """Synthetic OHLCV with mild drift + one mid-window drop to make bootstrap non-trivial."""
    import numpy as np
    rng = np.random.default_rng(7)
    rets = rng.normal(0.0003, 0.012, n_days)
    rets[120:135] = -0.02  # 15-day stretch with -2% daily moves
    closes = start_price * np.exp(np.cumsum(rets))
    idx = pd.date_range("2024-01-01", periods=n_days, freq="B")
    return pd.DataFrame(
        {
            "Open": closes, "High": closes * 1.005, "Low": closes * 0.995,
            "Close": closes, "Volume": [1_000_000] * n_days,
        },
        index=idx,
    )


def _stub_prices() -> PriceSnapshot:
    history = _stub_history()
    return PriceSnapshot(
        symbol="TEST", current_price=float(history["Close"].iloc[-1]),
        currency="USD", history=history,
        realized_vol_annualized_pct=30.0,
        cumulative_return_1m_pct=2.0,
        cumulative_return_3m_pct=-5.0,
        cumulative_return_1y_pct=8.0,
        timestamp_utc=datetime.now(timezone.utc),
    )


def _stub_fundamentals() -> FundamentalsSnapshot:
    return FundamentalsSnapshot(
        symbol="TEST", name="Test Corp",
        sector="Technology", industry="Software",
        market_cap=1_000_000_000, currency="USD",
        business_description="A software platform business.",
        pe_trailing=22.0, pe_forward=18.0, ev_to_ebitda=14.0,
        price_to_sales=5.0, price_to_book=3.5,
        gross_margin=0.7, operating_margin=0.2, profit_margin=0.15,
        return_on_equity=0.18, return_on_assets=0.10,
        debt_to_equity=0.4, current_ratio=1.6, quick_ratio=1.4,
        revenue_growth_yoy=0.18, earnings_growth_yoy=0.20,
        free_cash_flow=200_000_000, total_cash=500_000_000, total_debt=300_000_000,
        raw_info={"sharesOutstanding": 10_000_000, "trailingEps": 4.5, "forwardEps": 5.5},
    )


def _stub_macro() -> MacroSnapshot:
    return MacroSnapshot(
        yield_curve_10y3m_pct=0.5, yield_curve_10y2y_pct=0.3, vix_level=18.0,
        fed_funds_rate_pct=4.5, unemployment_rate_pct=4.0, cpi_yoy_pct=2.8,
        sp500_drawdown_pct=-3.0, hsi_drawdown_pct=-12.0,
        signals=[],
    )


def _stub_catalysts() -> CatalystSnapshot:
    return CatalystSnapshot(symbol="TEST", market="US")


def _stub_sentiment() -> SentimentSnapshot:
    return SentimentSnapshot(symbol="TEST", market="US")


def _stub_financials() -> FinancialReport:
    return FinancialReport(currency="USD", annual=None, quarterly=None, deep_resolution=None)


# ---------- chat_json dispatch ----------

def _stub_fundamental_assessment(**_kw) -> FundamentalAssessment:
    return FundamentalAssessment(
        quality_score=7.5, moat_assessment="strong network effects",
        moat_strength="wide", balance_sheet_health="strong",
        growth_outlook="durable double-digit", capital_allocation="disciplined buybacks",
        red_flags=[], thesis_one_liner="Quality software compounder.",
    )


def _stub_valuation_result(**_kw) -> ValuationResult:
    return ValuationResult(
        current_price=100.0, currency="USD",
        intrinsic_low=110.0, intrinsic_base=130.0, intrinsic_high=150.0,
        margin_of_safety_pct=30.0,
        methodology_notes="DCF + multiples blend",
        confidence="medium",
    )


def _stub_risk_assessment(**_kw) -> RiskAssessment:
    return RiskAssessment(
        scenarios=[Scenario(name="base", probability=1.0,
                            expected_return_pct=5, expected_drawdown_pct=-10,
                            rationale="benign")],
        drawdown_probabilities={"10": 0.30, "15": 0.15, "20": 0.08, "25": 0.04},
        realized_vol_annualized_pct=30.0,
        key_macro_signals=[],
        horizon_days=90,
    )


def _stub_forward_catalysts(**_kw) -> ForwardCatalysts:
    return ForwardCatalysts(symbol="TEST", horizon_days=30, key_catalysts=[])


def _stub_forward_scenarios(**_kw) -> ForwardScenarios:
    return ForwardScenarios(
        symbol="TEST", current_price=100.0, currency="USD", horizon_days=90,
        scenarios=[
            PriceScenario(name="base", probability=1.0,
                          target_price_low=95, target_price_base=105, target_price_high=115,
                          return_pct_base=5.0, drawdown_pct_estimated=-10.0,
                          key_drivers=["base"], rationale="base case"),
        ],
    )


def _stub_contrarian(**_kw) -> ContrarianAssessment:
    return ContrarianAssessment(
        crowd_position="neutral", contrarian_signal="neutral",
        reasoning="neutral positioning", data_quality="medium",
    )


def _stub_price_map(**_kw) -> PriceMap:
    return PriceMap(
        zones=[PriceMapZone(price_low=90, price_high=110, label="hold", rationale="stub")],
        current_zone_index=0, key_support=92.0, key_resistance=115.0,
        summary="stub map",
    )


def _stub_hedge_plan(**_kw) -> HedgePlan:
    return HedgePlan(
        symbol_being_hedged="TEST", position_value_usd=100_000.0,
        candidates=[], recommended_index=0, rationale="no hedge needed",
    )


def _stub_devil_review_pass(**_kw) -> DevilAdvocateReview:
    return DevilAdvocateReview(
        overall_verdict="pass", summary="clean",
        findings=[], counter_thesis="bears would struggle", veto_reason=None,
    )


def _stub_devil_review_veto(**_kw) -> DevilAdvocateReview:
    return DevilAdvocateReview(
        overall_verdict="veto", summary="material flaw",
        findings=[DevilFinding(category="moat_erosion", severity="veto",
                               finding="moat broken", evidence="evidence",
                               recommendation="abandon")],
        counter_thesis="bear case is moat erosion",
        veto_reason="Moat is broken — revenue concentration risk.",
    )


# Map schema-class-name → factory. The integration test selects the DA factory
# per test via a function attribute on this dispatcher.
_SCHEMA_DISPATCH = {
    "FundamentalAssessment": _stub_fundamental_assessment,
    "ValuationResult": _stub_valuation_result,
    "RiskAssessment": _stub_risk_assessment,
    "ForwardCatalysts": _stub_forward_catalysts,
    "ForwardScenarios": _stub_forward_scenarios,
    "ContrarianAssessment": _stub_contrarian,
    "PriceMap": _stub_price_map,
    "HedgePlan": _stub_hedge_plan,
    # DevilAdvocateReview handled by per-test override
}


def _make_chat_json_stub(devil_factory):
    """Returns a fake chat_json that dispatches on schema kwarg."""
    def fake_chat_json(*, system, user, schema, **kw):
        name = schema.__name__
        if name == "DevilAdvocateReview":
            return devil_factory()
        factory = _SCHEMA_DISPATCH.get(name)
        if factory is None:
            raise RuntimeError(f"integration test missing stub for schema {name}")
        return factory()
    return fake_chat_json


# ---------- Fixture ----------

@pytest.fixture
def patched_pipeline(monkeypatch, request):
    """Patches all upstream IO + LLM calls so analyze() runs without network.

    The `request.param` selects the Devil's Advocate verdict:
      "pass" → _stub_devil_review_pass
      "veto" → _stub_devil_review_veto
    """
    devil_factory = _stub_devil_review_pass
    if hasattr(request, "param") and request.param == "veto":
        devil_factory = _stub_devil_review_veto

    # Patch data-fetch entry points as the orchestrator imported them
    from src.pipeline import orchestrator
    monkeypatch.setattr(orchestrator, "fetch_prices", lambda *a, **kw: _stub_prices())
    monkeypatch.setattr(orchestrator, "fetch_fundamentals", lambda *a, **kw: _stub_fundamentals())
    monkeypatch.setattr(orchestrator, "fetch_catalysts", lambda *a, **kw: _stub_catalysts())
    monkeypatch.setattr(orchestrator, "fetch_sentiment", lambda *a, **kw: _stub_sentiment())
    monkeypatch.setattr(orchestrator, "fetch_financials", lambda *a, **kw: _stub_financials())
    # Macro is wrapped in @lru_cache — clear it and patch the wrapped fetcher
    monkeypatch.setattr(orchestrator, "fetch_macro", lambda *a, **kw: _stub_macro())
    orchestrator.reset_macro_cache()

    # Patch chat_json on every agent module
    stub = _make_chat_json_stub(devil_factory)
    for module_name in (
        "src.agents.fundamental",
        "src.agents.valuation",
        "src.agents.risk_analyzer",
        "src.agents.financial_report",
        "src.agents.information_retrieval",
        "src.agents.forward_scenario",
        "src.agents.contrarian",
        "src.agents.price_map",
        "src.agents.hedging",
        "src.agents.devil_advocate",
    ):
        try:
            mod = __import__(module_name, fromlist=["chat_json"])
            if hasattr(mod, "chat_json"):
                monkeypatch.setattr(mod, "chat_json", stub)
        except ImportError:
            pass

    yield orchestrator


# ---------- Tests ----------

def test_pipeline_completes_end_to_end(patched_pipeline):
    """Wiring contract: analyze() runs without raising and returns a valid result."""
    result = patched_pipeline.analyze("TEST", "US", persist=False)
    assert result.symbol == "TEST"
    assert result.market == "US"
    assert result.current_price > 0
    assert result.fundamental.quality_score == 7.5
    assert result.valuation.intrinsic_base == 130.0
    assert result.risk.horizon_days == 90
    assert result.if_held is not None
    assert result.if_not_held is not None
    # DA passes → BUY_NOW survives
    assert result.if_not_held.recommendation == "BUY_NOW"
    assert len(result.if_not_held.entry_orders) >= 1


def test_bootstrap_drawdown_prior_is_attached(patched_pipeline):
    """Bootstrap sanity check should populate `risk.bootstrap_drawdown_probabilities`
    because our stub history has enough bars + a synthetic drawdown stretch."""
    result = patched_pipeline.analyze("TEST", "US", persist=False)
    assert result.risk.bootstrap_drawdown_probabilities is not None
    # Bootstrap returns the standard 4 buckets
    for k in ("10", "15", "20", "25"):
        assert k in result.risk.bootstrap_drawdown_probabilities


@pytest.mark.parametrize("patched_pipeline", ["veto"], indirect=True)
def test_devil_veto_clears_orders_in_full_pipeline(patched_pipeline):
    """The project docs claim DA can VETO. This test enforces that claim.

    With the DA stub returning verdict=veto, the orchestrator's apply_devil_veto
    must run and:
      * blank `if_held.immediate_orders` and `rebuy_orders`
      * force `if_not_held.recommendation` to PASS
      * stamp the rationale with the veto marker
    """
    result = patched_pipeline.analyze("TEST", "US", persist=False)

    assert result.devil_advocate is not None
    assert result.devil_advocate.overall_verdict == "veto"

    # If-held side
    assert result.if_held.immediate_orders == []
    assert result.if_held.rebuy_orders == []
    assert "DEVIL'S ADVOCATE VETO" in result.if_held.tactical.rationale

    # If-not-held side
    assert result.if_not_held.recommendation == "PASS"
    assert result.if_not_held.entry_orders == []
    assert "DEVIL'S ADVOCATE VETO" in result.if_not_held.rationale


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
