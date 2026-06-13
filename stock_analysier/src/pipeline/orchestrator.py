"""Orchestrator — runs the full pipeline for a single ticker.

Pipeline:
  1. Fetch data: prices, fundamentals, macro (macro cached per session)
  2. Run Fundamental Analyst (LLM)
  3. Run Valuation agent (LLM)
  4. Run Risk Analyzer (LLM)
  5. Always compute BOTH paths:
       - if_held: Tactical Exit → Order Generator
       - if_not_held: valuation-based entry decision (Order Generator)
  6. Return AnalysisResult
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from functools import lru_cache
from typing import Callable, Literal, Optional

# Progress callback: (step_name, status, optional_message)
# step_name ∈ a small fixed vocabulary defined below
# status    ∈ "started" | "completed" | "skipped" | "failed"
ProgressCallback = Callable[[str, str, Optional[str]], None]

# All possible pipeline steps, in execution order. Used by the UI to render
# a placeholder list before any events fire.
PIPELINE_STEPS = [
    "data_fetch",         # prices + fundamentals + macro
    "competence_gate",
    "fundamental",
    "valuation",
    "financial_report",   # last-2-period statements snapshot + LLM deep resolution
    "catalysts_data",
    "ir_agent",
    "forward_scenarios",
    "risk_analyzer",
    # Technical Division (blinded by signature — sees prices/volume only).
    # Runs after Valuation in time so Price Map can use intrinsic price points,
    # but agents NEVER receive fundamental/valuation/macro objects.
    "structure",
    "volume_flow",
    "cost_basis",
    "relative_strength",
    "price_map",
    "technical_composite",
    # Downstream consumers
    "contrarian",
    "tactical_exit",
    "order_generator",
    "hedging",
    "devil_advocate",
    "audit_persist",
]

from src.agents.competence_gate import assess as competence_assess, skip_pipeline_for
from src.agents.contrarian import assess_contrarian
from src.agents.cost_basis import assess_cost_basis
from src.agents.devil_advocate import apply_veto as apply_devil_veto
from src.agents.devil_advocate import review as run_devil_advocate
from src.agents.financial_report import analyze_financials
from src.agents.forward_scenario import generate_scenarios as generate_forward_scenarios
from src.agents.fundamental import analyze_fundamentals
from src.agents.hedging import design_hedge
from src.agents.information_retrieval import analyze_catalysts
from src.agents.order_generator import generate_entry_orders, generate_held_orders
from src.agents.price_map import build_price_map
from src.agents.relative_strength import assess_relative_strength
from src.agents.risk_analyzer import analyze_risk
from src.agents.structure import assess_structure
from src.agents.tactical_exit import decide_tactical
from src.agents.technical_director import assemble as assemble_technical
from src.agents.valuation import analyze_valuation
from src.agents.volume_flow import assess_volume
from src.config.loader import load_portfolio, load_risk_policy, load_technical
from src.data.catalysts import fetch_catalysts
from src.data.financials import fetch_financials
from src.data.fundamentals import fetch_fundamentals
from src.data.macro import MacroSnapshot, fetch_macro
from src.data.prices import fetch_prices
from src.data.sentiment import fetch_sentiment
from src.models.schemas import AnalysisResult, Holding, TechnicalAssessment
from src.storage.audit import get_latest_run, record_analysis

log = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _cached_macro() -> MacroSnapshot:
    log.info("Fetching macro snapshot (cached for session)…")
    return fetch_macro()


def reset_macro_cache() -> None:
    _cached_macro.cache_clear()


def _prior_audit_summary(symbol: str) -> Optional[str]:
    """Build a 1-line summary of the most recent prior run for Devil's Advocate context."""
    row = get_latest_run(symbol)
    if row is None:
        return None
    return (
        f"Prior run at {row['timestamp_utc']}: "
        f"price={row['current_price']}, quality={row['quality_score']:.1f}, "
        f"MoS={row['margin_of_safety_pct']:.1f}%, "
        f"P(dd≥15%)={row['p_dd_15']:.2f}, P(dd≥20%)={row['p_dd_20']:.2f}, "
        f"tactical={row['tactical_label'] or 'NO_ACTION'}, "
        f"if_not_held={row['if_not_held_recommendation']}, "
        f"DA verdict={row['devil_verdict'] or 'n/a'}."
    )


def analyze(
    symbol: str,
    market: Literal["US", "HK"],
    available_cash_usd: Optional[float] = None,
    skip_catalysts: bool = False,
    skip_devil_advocate: bool = False,
    persist: bool = True,
    on_progress: Optional[ProgressCallback] = None,
) -> AnalysisResult:
    log.info("Analyzing %s (%s)…", symbol, market)

    def _progress(step: str, status: str, msg: Optional[str] = None) -> None:
        if on_progress is not None:
            try:
                on_progress(step, status, msg)
            except Exception as e:  # never let a progress reporter break the pipeline
                log.debug("progress callback failed: %s", e)

    # 1. Data
    _progress("data_fetch", "started")
    try:
        prices = fetch_prices(symbol, period="1y", vol_window_days=30)
        fundamentals = fetch_fundamentals(symbol)
        macro = _cached_macro()
        _progress("data_fetch", "completed")
    except Exception as e:
        _progress("data_fetch", "failed", str(e))
        raise
    portfolio = load_portfolio()
    risk_policy = load_risk_policy()
    holding: Optional[Holding] = portfolio.find(symbol)

    # 1b. Competence gate — quick deterministic pre-filter
    _progress("competence_gate", "started")
    competence = competence_assess(symbol, fundamentals.sector, fundamentals.business_description)
    _progress("competence_gate", "completed", competence.verdict)
    if skip_pipeline_for(competence):
        log.info("Skipping %s — outside circle of competence per policy.", symbol)
        for step in [
            "fundamental", "valuation", "financial_report",
            "catalysts_data", "ir_agent",
            "forward_scenarios", "risk_analyzer",
            "structure", "volume_flow", "cost_basis", "relative_strength",
            "price_map", "technical_composite",
            "tactical_exit", "order_generator", "hedging",
            "contrarian", "devil_advocate", "audit_persist",
        ]:
            _progress(step, "skipped", "outside circle of competence")
        # Return a minimal AnalysisResult to keep downstream code happy
        from src.models.schemas import (
            FundamentalAssessment, HeldDecision, NotHeldDecision,
            RiskAssessment, Scenario, TacticalAction, ValuationResult,
        )
        return AnalysisResult(
            symbol=symbol, market=market,
            timestamp_utc=datetime.now(timezone.utc),
            current_price=prices.current_price, currency=prices.currency,
            position=holding,
            fundamental=FundamentalAssessment(
                quality_score=0, moat_assessment="not assessed",
                moat_strength="none", balance_sheet_health="adequate",
                growth_outlook="not assessed", capital_allocation="not assessed",
                red_flags=["Outside circle of competence"],
                thesis_one_liner="Skipped per competence policy.",
            ),
            valuation=ValuationResult(
                current_price=prices.current_price, currency=prices.currency,
                intrinsic_low=0, intrinsic_base=0, intrinsic_high=0,
                margin_of_safety_pct=0, methodology_notes="not assessed", confidence="low",
            ),
            risk=RiskAssessment(
                scenarios=[Scenario(name="base", probability=1.0, expected_return_pct=0, expected_drawdown_pct=0, rationale="not assessed")],
                drawdown_probabilities={"10": 0, "15": 0, "20": 0, "25": 0},
                realized_vol_annualized_pct=prices.realized_vol_annualized_pct,
                key_macro_signals=[], horizon_days=risk_policy.forecast_horizon_days,
            ),
            if_held=HeldDecision(
                tactical=TacticalAction(action="no_action", rationale="Outside circle of competence — no pipeline run."),
            ),
            if_not_held=NotHeldDecision(
                recommendation="PASS",
                rationale=f"Outside circle of competence: {competence.reasoning}",
            ),
            competence=competence,
        )

    # 2. Fundamental + Valuation (LLM)
    _progress("fundamental", "started")
    try:
        fundamental = analyze_fundamentals(fundamentals)
        _progress("fundamental", "completed", f"q={fundamental.quality_score:.1f}/10 moat={fundamental.moat_strength}")
    except Exception as e:
        _progress("fundamental", "failed", str(e))
        raise

    _progress("valuation", "started")
    try:
        valuation = analyze_valuation(fundamentals, prices, fundamental, market=market)
        _progress("valuation", "completed", f"MoS {valuation.margin_of_safety_pct:+.1f}%")
    except Exception as e:
        _progress("valuation", "failed", str(e))
        raise

    # 2b. Financial Report — fetch last-2-period statements + deep LLM resolution
    financial_report = None
    _progress("financial_report", "started")
    try:
        financial_report = fetch_financials(symbol)
        try:
            financial_report.deep_resolution = analyze_financials(
                symbol=symbol,
                name=fundamentals.name,
                sector=fundamentals.sector,
                report=financial_report,
            )
            ann = len(financial_report.annual.periods) if financial_report.annual else 0
            q = len(financial_report.quarterly.periods) if financial_report.quarterly else 0
            _progress("financial_report", "completed", f"{ann} annual + {q} quarterly periods, deep resolution generated")
        except Exception as e:
            log.warning("Financial deep-resolution LLM failed for %s: %s — returning raw snapshot only", symbol, e)
            _progress("financial_report", "completed", "raw snapshot only — LLM resolver failed")
    except Exception as e:
        log.warning("Financial Report fetch failed for %s: %s — continuing", symbol, e)
        _progress("financial_report", "failed", str(e))

    # 3. Forward catalysts (IR agent)
    forward_catalysts = None
    if not skip_catalysts:
        _progress("catalysts_data", "started")
        try:
            catalysts_raw = fetch_catalysts(
                symbol=symbol,
                market=market,
                name_hint=fundamentals.name,
                horizon_days=30,
                news_lookback_days=14,
            )
            _progress("catalysts_data", "completed",
                      f"earnings={len(catalysts_raw.upcoming_earnings)} news={len(catalysts_raw.recent_news)} econ={len(catalysts_raw.upcoming_economic_events)}")
        except Exception as e:
            _progress("catalysts_data", "failed", str(e))
            catalysts_raw = None
            log.warning("Catalyst fetch failed for %s: %s", symbol, e)

        if catalysts_raw is not None:
            _progress("ir_agent", "started")
            try:
                forward_catalysts = analyze_catalysts(
                    catalysts_raw,
                    sector=fundamentals.sector,
                    business_summary=fundamentals.business_description,
                )
                _progress("ir_agent", "completed",
                          f"{len(forward_catalysts.key_catalysts)} catalysts, sentiment {forward_catalysts.sentiment_score:+.2f}")
            except Exception as e:
                _progress("ir_agent", "failed", str(e))
                log.warning("IR agent failed for %s: %s — continuing without forward catalysts", symbol, e)
    else:
        _progress("catalysts_data", "skipped")
        _progress("ir_agent", "skipped")

    # 3b. Forward Scenario agent (probability-weighted price paths at horizon)
    forward_scenarios = None
    _progress("forward_scenarios", "started")
    try:
        forward_scenarios = generate_forward_scenarios(
            symbol=symbol,
            current_price=prices.current_price,
            currency=prices.currency,
            valuation=valuation,
            macro=macro,
            horizon_days=risk_policy.forecast_horizon_days,
            forward_catalysts=forward_catalysts,
        )
        _progress("forward_scenarios", "completed",
                  f"{len(forward_scenarios.scenarios)} scenarios, E[ret]={forward_scenarios.expected_return_pct:+.1f}%")
    except Exception as e:
        _progress("forward_scenarios", "failed", str(e))
        log.warning("Forward Scenario agent failed for %s: %s — continuing", symbol, e)

    # 4. Risk Analyzer (consumes forward catalysts)
    _progress("risk_analyzer", "started")
    try:
        risk = analyze_risk(
            symbol=symbol,
            p=prices,
            f=fundamentals,
            fa=fundamental,
            v=valuation,
            m=macro,
            horizon_days=risk_policy.forecast_horizon_days,
            forward_catalysts=forward_catalysts,
        )
        _progress("risk_analyzer", "completed",
                  f"P(dd≥15%)={risk.drawdown_probabilities.get('15', 0):.2f}, vol={risk.realized_vol_annualized_pct:.0f}%")
    except Exception as e:
        _progress("risk_analyzer", "failed", str(e))
        raise

    # 4b. TECHNICAL DIVISION (Phase 6) — blinded by signature: each agent receives
    # ONLY prices/volume/sector-string/current-price. Even though they execute in
    # this position chronologically, no fundamental/valuation/macro object can leak in.
    technical_assessment: Optional[TechnicalAssessment] = None
    try:
        tech_cfg = load_technical()
        _progress("structure", "started")
        try:
            structure = assess_structure(prices.history, tech_cfg.structure)
            _progress("structure", "completed", f"{structure.trend} ({structure.stage})")
        except Exception as e:
            _progress("structure", "failed", str(e))
            raise

        _progress("volume_flow", "started")
        try:
            volume = assess_volume(prices.history, tech_cfg.volume)
            _progress("volume_flow", "completed",
                      f"{volume.institutional_flow} / OBV {volume.obv_trend}")
        except Exception as e:
            _progress("volume_flow", "failed", str(e))
            raise

        _progress("cost_basis", "started")
        try:
            cost_basis = assess_cost_basis(prices.history, tech_cfg.cost_basis, prices.current_price)
            _progress("cost_basis", "completed",
                      f"{len(cost_basis.hvn_levels)} HVN, trapped {cost_basis.trapped_supply_pct:.0f}%")
        except Exception as e:
            _progress("cost_basis", "failed", str(e))
            raise

        _progress("relative_strength", "started")
        try:
            ticker_returns = {
                90: prices.cumulative_return_3m_pct if prices.cumulative_return_3m_pct == prices.cumulative_return_3m_pct else None,
                365: prices.cumulative_return_1y_pct if prices.cumulative_return_1y_pct == prices.cumulative_return_1y_pct else None,
            }
            rel_strength = assess_relative_strength(
                symbol=symbol, market=market, sector=fundamentals.sector,
                ticker_returns=ticker_returns, cfg=tech_cfg.relative_strength,
            )
            _progress("relative_strength", "completed", rel_strength.signal)
        except Exception as e:
            _progress("relative_strength", "failed", str(e))
            raise

        _progress("price_map", "started")
        try:
            price_map = build_price_map(
                symbol=symbol, current_price=prices.current_price, currency=prices.currency,
                intrinsic_low=valuation.intrinsic_low,
                intrinsic_base=valuation.intrinsic_base,
                intrinsic_high=valuation.intrinsic_high,
                structure=structure, volume=volume,
                cost_basis=cost_basis, relative_strength=rel_strength,
                enable_llm=tech_cfg.price_map.enable,
            )
            _progress("price_map", "completed",
                      f"{len(price_map.zones)} zones, key_support={price_map.key_support}")
        except Exception as e:
            _progress("price_map", "failed", str(e))
            raise

        _progress("technical_composite", "started")
        technical_assessment = assemble_technical(
            structure=structure, volume=volume, cost_basis=cost_basis,
            relative_strength=rel_strength, price_map=price_map,
        )
        _progress("technical_composite", "completed", technical_assessment.composite_signal)
    except Exception as e:
        log.warning("Technical Division failed for %s: %s — continuing without it", symbol, e)
        technical_assessment = None

    # 5a. If held path (always computed; UI decides which to show)
    _progress("tactical_exit", "started")
    if holding is not None:
        holding_for_orders = holding
    else:
        holding_for_orders = Holding(
            symbol=symbol,
            market=market,
            shares=0,
            cost_basis_per_share=prices.current_price,
            currency=prices.currency,
        )
    tactical = decide_tactical(holding_for_orders, prices.current_price, risk, risk_policy, technical=technical_assessment)
    _progress("tactical_exit", "completed", tactical.label or "no_action")

    _progress("order_generator", "started")
    held_decision = generate_held_orders(holding_for_orders, prices.current_price, tactical)
    _progress("order_generator", "completed",
              f"{len(held_decision.immediate_orders)} immediate + {len(held_decision.rebuy_orders)} rebuy")

    # 5c. Hedging plan (only when Tactical Exit recommends a hedge AND position exists with value)
    hedge_plan = None
    if (
        holding is not None
        and held_decision.tactical.hedge_recommended
        and holding.shares > 0
    ):
        from src.data.fx import convert as fx_convert
        pos_value_local = holding.shares * prices.current_price
        pos_value_usd = fx_convert(pos_value_local, prices.currency, "USD")
        min_usd = risk_policy.hedging_minimums.min_position_value_usd
        if pos_value_usd < min_usd:
            _progress(
                "hedging", "skipped",
                f"position ${pos_value_usd:,.0f} below min hedge size ${min_usd:,.0f}",
            )
            log.info(
                "Skipping hedge for %s — $%.0f under hedging_minimums.min_position_value_usd $%.0f.",
                symbol, pos_value_usd, min_usd,
            )
        else:
            _progress("hedging", "started")
            try:
                hedge_plan = design_hedge(
                    symbol=symbol,
                    market=market,
                    sector=fundamentals.sector,
                    position_value_usd=pos_value_usd,
                    prefer_etf_short_below_usd=risk_policy.hedging_minimums.prefer_etf_short_when_below_usd,
                )
                _progress("hedging", "completed",
                          f"{len(hedge_plan.candidates)} candidates" if hedge_plan else "no hedge")
            except Exception as e:
                _progress("hedging", "failed", str(e))
                log.warning("Hedging agent failed for %s: %s", symbol, e)
    else:
        _progress("hedging", "skipped", "not triggered by tactical exit")

    # 5b. If not held path
    not_held_decision = generate_entry_orders(
        symbol=symbol,
        market=market,
        current_price=prices.current_price,
        valuation=valuation,
        available_cash=available_cash_usd,
    )

    # 5d. Contrarian / sentiment agent (LLM; best-effort, US-only data on free tier)
    contrarian_assessment = None
    _progress("contrarian", "started")
    try:
        sentiment_snap = fetch_sentiment(symbol, market)
        contrarian_assessment = assess_contrarian(
            symbol=symbol, market=market,
            sentiment=sentiment_snap, prices=prices, valuation=valuation,
        )
        _progress("contrarian", "completed",
                  f"{contrarian_assessment.crowd_position}/{contrarian_assessment.contrarian_signal}")
    except Exception as e:
        _progress("contrarian", "failed", str(e))
        log.warning("Contrarian agent failed for %s: %s — continuing", symbol, e)

    # If competence verdict was 'out_of_circle' (but policy = analyze_but_flag), downgrade
    # the not-held recommendation to PASS so the user has to consciously override.
    if competence.verdict == "out_of_circle" and not_held_decision.recommendation == "BUY_NOW":
        not_held_decision.recommendation = "PASS"
        not_held_decision.rationale = (
            f"OUTSIDE CIRCLE OF COMPETENCE — original recommendation was BUY_NOW, downgraded to PASS. "
            f"Reason: {competence.reasoning}"
        )

    result = AnalysisResult(
        symbol=symbol,
        market=market,
        timestamp_utc=datetime.now(timezone.utc),
        current_price=prices.current_price,
        currency=prices.currency,
        position=holding,
        fundamental=fundamental,
        valuation=valuation,
        risk=risk,
        if_held=held_decision,
        if_not_held=not_held_decision,
        forward_catalysts=forward_catalysts,
        forward_scenarios=forward_scenarios,
        hedge_plan=hedge_plan,
        competence=competence,
        contrarian=contrarian_assessment,
        technical=technical_assessment,
        financial_report=financial_report,
    )

    # 6. Devil's Advocate review — reads everything above and tries to break it.
    #    On a veto verdict, `apply_devil_veto` strips orders so the dashboard /
    #    Telegram / audit record reflect the veto, not the pre-veto orders.
    if not skip_devil_advocate:
        _progress("devil_advocate", "started")
        try:
            prior_summary = _prior_audit_summary(symbol)
            result.devil_advocate = run_devil_advocate(result, prior_audit_summary=prior_summary)
            vetoed = apply_devil_veto(result, result.devil_advocate)
            _progress(
                "devil_advocate", "completed",
                f"{result.devil_advocate.overall_verdict} "
                f"({len(result.devil_advocate.findings)} findings"
                f"{', ORDERS CLEARED' if vetoed else ''})",
            )
            log.info(
                "Devil's Advocate verdict for %s: %s (%d findings, vetoed=%s)",
                symbol, result.devil_advocate.overall_verdict,
                len(result.devil_advocate.findings), vetoed,
            )
        except Exception as e:
            _progress("devil_advocate", "failed", str(e))
            log.warning("Devil's Advocate failed for %s: %s — continuing", symbol, e)
            result.errors.append(f"Devil's Advocate failed: {e}")
    else:
        _progress("devil_advocate", "skipped")

    # 7. Persist to audit trail
    if persist:
        _progress("audit_persist", "started")
        try:
            record_analysis(result)
            _progress("audit_persist", "completed")
        except Exception as e:
            _progress("audit_persist", "failed", str(e))
            log.warning("Audit trail write failed for %s: %s", symbol, e)
            result.errors.append(f"Audit trail write failed: {e}")
    else:
        _progress("audit_persist", "skipped")

    return result


def analyze_all() -> list[AnalysisResult]:
    """Run analysis on every holding + every watchlist entry across active markets."""
    from src.config.loader import load_universe

    portfolio = load_portfolio()
    universe = load_universe()

    seen = set()
    targets: list[tuple[str, str]] = []

    for h in portfolio.holdings:
        if h.symbol not in seen:
            targets.append((h.symbol, h.market))
            seen.add(h.symbol)

    for market in universe.active_markets:
        for w in universe.watchlist.get(market, []):
            if w.symbol not in seen:
                targets.append((w.symbol, market))
                seen.add(w.symbol)

    results: list[AnalysisResult] = []
    for sym, mkt in targets:
        try:
            results.append(analyze(sym, mkt))
        except Exception as e:
            log.exception("Pipeline failed for %s: %s", sym, e)
    return results
