"""Unit tests for the LLM-free pre-market overview (``digest.build_overview_digest``).

Pure string assembly — no network, no LLM.
"""

from __future__ import annotations

import pytest

from tradingagents.runtime.digest import build_overview_digest

pytestmark = pytest.mark.unit

BOOK = {"held": [
    {"symbol": "FIG", "cost_basis_per_share": 55.0},
    {"symbol": "0700.HK", "cost_basis_per_share": 500.0},
]}
MACRO = {"risk_level": "HIGH", "tally": {"red": 2, "yellow": 1, "green": 3}}


def test_holdings_show_pnl_vs_cost():
    out = build_overview_digest("2026-06-17", MACRO, BOOK,
                                {"FIG": 50.0, "0700.HK": 520.0}, [], None, [])
    assert "FIG: -9% vs cost" in out
    assert "0700.HK: +4% vs cost" in out


def test_missing_price_renders_na_not_crash():
    out = build_overview_digest("2026-06-17", MACRO, BOOK, {}, [], None, [])
    assert "FIG: n/a" in out


def test_alarms_listed_with_reasons():
    alarms = [{"symbol": "FIG", "reasons": ["-9% vs cost", "risk-off"]}]
    out = build_overview_digest("2026-06-17", MACRO, BOOK,
                                {"FIG": 50.0}, [], "Risk-Off Defensive", alarms)
    assert "⚠ FIG: -9% vs cost, risk-off" in out
    assert "auto-runs" in out


def test_no_alarms_prompts_pull():
    out = build_overview_digest("2026-06-17", MACRO, BOOK, {"FIG": 60.0}, [], None, [])
    assert "Nothing tripped" in out
    assert "/us TICKER" in out


def test_ranking_capped_at_six():
    ranking = [{"symbol": f"T{i}", "rs_score": float(i), "rank": i} for i in range(1, 11)]
    out = build_overview_digest("2026-06-17", MACRO, {"held": []},
                                {}, ranking, None, [])
    assert "1. T1" in out and "6. T6" in out
    assert "7. T7" not in out


def test_regime_and_macro_action_lines():
    macro = {**MACRO, "action": "Trim risk; raise cash."}
    out = build_overview_digest("2026-06-17", macro, {"held": []},
                                {}, [], "Risk-On", [])
    assert "Regime: **Risk-On**" in out
    assert "Trim risk; raise cash." in out


def test_no_macro_is_graceful():
    out = build_overview_digest("2026-06-17", None, {"held": []}, {}, [], None, [])
    assert "Market context: unavailable" in out


# ----------------------- dedupe-aware rendering ------------------------------

def test_running_syms_marks_run_vs_covered():
    alarms = [
        {"symbol": "FIG", "reasons": ["-20% from high"]},
        {"symbol": "0700.HK", "reasons": ["-19% from high"]},
    ]
    out = build_overview_digest("2026-06-17", MACRO, BOOK,
                                {"FIG": 50.0, "0700.HK": 400.0}, [], None,
                                alarms, running_syms=["FIG"])
    assert "▶ FIG:" in out
    assert "⏸ 0700.HK:" in out
    assert "deep-diving now" in out


def test_running_syms_empty_means_all_covered():
    alarms = [{"symbol": "FIG", "reasons": ["-20% from high"]}]
    out = build_overview_digest("2026-06-17", MACRO, BOOK, {"FIG": 50.0}, [], None,
                                alarms, running_syms=[])
    assert "⏸ FIG:" in out
    assert "already covered" in out


def test_legacy_rendering_when_running_syms_none():
    # Back-compat: no dedupe info -> original ⚠ rendering.
    alarms = [{"symbol": "FIG", "reasons": ["-20% from high"]}]
    out = build_overview_digest("2026-06-17", MACRO, BOOK, {"FIG": 50.0}, [], None, alarms)
    assert "⚠ FIG:" in out
    assert "auto-runs" in out
