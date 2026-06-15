"""Unit tests for the position-aware overlay (Phase 1) + danger gate (Phase 2.1).

Pure functions, no I/O — mirrors the orchestrator/tests style for rules.py.
"""

from __future__ import annotations

from datetime import date

import pytest

from tradingagents.portfolio.position_overlay import (
    OverlayConfig,
    Position,
    evaluate,
    is_danger,
    needs_action,
)

pytestmark = pytest.mark.unit

CFG = OverlayConfig()


def _held(rating, **kw):
    return Position(symbol="X", rating=rating, held=True, **kw)


def _watch(rating, **kw):
    return Position(symbol="X", rating=rating, held=False, **kw)


# ------------------------------- base mapping --------------------------------

def test_held_buy_holds():
    assert evaluate(_held("Buy"))["action"] == "HOLD"


def test_held_sell_defensive():
    assert evaluate(_held("Sell"))["action"] == "DEFENSIVE"


def test_held_underweight_trims():
    assert evaluate(_held("Underweight"))["action"] == "TRIM"


def test_watch_buy_opens():
    assert evaluate(_watch("Buy"))["action"] == "BUY_NOW"


def test_watch_sell_avoids():
    assert evaluate(_watch("Sell"))["action"] == "AVOID"


# ------------------------------- concentration -------------------------------

def test_hard_concentration_forces_trim_even_on_buy():
    v = evaluate(_held("Buy", weight_pct=55.0))
    assert v["action"] == "TRIM"
    assert any("hard cap" in n for n in v["notes"])


def test_soft_concentration_trims_only_when_not_bullish():
    # Hold at 30% -> trim; Buy at 30% -> still HOLD (let winner run below hard cap).
    assert evaluate(_held("Hold", weight_pct=30.0))["action"] == "TRIM"
    assert evaluate(_held("Buy", weight_pct=30.0))["action"] == "HOLD"


# --------------------------- profit / stop discipline ------------------------

def test_big_gain_hardens_bearish_call():
    base = evaluate(_held("Underweight", cost_basis=10, current_price=10))["action"]
    hardened = evaluate(_held("Underweight", cost_basis=10, current_price=25))["action"]
    assert base == "TRIM"
    assert hardened == "DEFENSIVE"  # escalated one rung by the +150% gain


def test_stop_loss_on_sell_exits():
    v = evaluate(_held("Sell", cost_basis=100, current_price=70))  # -30%
    assert v["action"] == "EXIT"
    assert any("stop" in n for n in v["notes"])


# ------------------------------- danger gate ---------------------------------

def test_is_danger_macro_and_regime():
    assert is_danger("HIGH", None, CFG)[0] is True
    assert is_danger(None, "Risk-Off Defensive", CFG)[0] is True
    assert is_danger("LOW", "Risk-On", CFG)[0] is False


def test_danger_escalates_held_one_rung():
    calm = evaluate(_held("Hold"))["action"]            # WATCH
    danger = evaluate(_held("Hold"), macro_level="CRITICAL")
    assert calm == "WATCH"
    assert danger["action"] == "TRIM"
    assert danger["escalated"] is True


def test_danger_blocks_new_entry():
    v = evaluate(_watch("Buy"), regime="Deflationary Shock")
    assert v["action"] == "HOLD_OFF"
    assert v["entry_blocked"] is True


def test_calm_is_not_a_ratchet():
    # Symmetry check: with no danger, the action equals the un-escalated base.
    v = evaluate(_held("Hold"), macro_level="LOW", regime="Risk-On")
    assert v["action"] == "WATCH"
    assert v["escalated"] is False


# ------------------------------- tax note ------------------------------------

def test_short_term_tax_note_on_us_trim():
    v = evaluate(_held("Underweight", market="US", purchase_date="2099-01-01"),
                 today=date(2099, 3, 1))  # ~59 days held
    assert any("short-term" in n for n in v["notes"])


# ------------------------------- needs_action --------------------------------

def test_needs_action():
    assert needs_action(evaluate(_held("Sell"))) is True
    assert needs_action(evaluate(_held("Buy"))) is False
    assert needs_action(evaluate(_watch("Buy"))) is True
    assert needs_action(evaluate(_watch("Hold"))) is False


def test_bad_rating_rejected():
    with pytest.raises(ValueError):
        Position(symbol="X", rating="StrongBuy", held=True)
