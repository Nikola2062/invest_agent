"""Unit tests for the launcher runtime: digest, bot command parsing, positions
overlay integration, outcome logging. No network, no LLM (prices/FX monkeypatched).
"""

from __future__ import annotations

import pytest

from tradingagents.runtime import digest as digest_mod
from tradingagents.runtime import bot
from tradingagents.runtime import reflect
from tradingagents.portfolio import positions as positions_mod

pytestmark = pytest.mark.unit


# ------------------------------- digest --------------------------------------

def _verdicts():
    return [
        {"symbol": "FIG", "held": True, "rating": "Underweight", "base_action": "TRIM",
         "action": "TRIM", "escalated": False, "unrealized_pnl_pct": 30.0, "weight_pct": 20.0},
        {"symbol": "NVDA", "held": False, "rating": "Buy", "base_action": "BUY_NOW",
         "action": "BUY_NOW", "entry_blocked": False},
        {"symbol": "AMD", "held": False, "rating": "Hold", "base_action": "HOLD_OFF",
         "action": "HOLD_OFF", "entry_blocked": False},
    ]


def test_digest_contains_sections():
    macro = {"risk_level": "MEDIUM", "tally": {"red": 2, "yellow": 3, "green": 3},
             "sp500": {"drawdown_pct": -1.6}, "action": "HOLD — do not deploy cash."}
    ranking = [{"symbol": "NVDA", "rs_score": 8.2, "rank": 1},
               {"symbol": "AMD", "rs_score": -1.0, "rank": 2}]
    out = digest_mod.build_digest("2026-06-15", macro, _verdicts(), ranking, regime="Risk-On")
    for token in ["Pre-market digest", "MEDIUM", "Your holdings", "FIG", "Opportunity rank",
                  "Action summary", "Risk-On"]:
        assert token in out
    # FIG (TRIM) needs action, NVDA (BUY_NOW) needs action, AMD (HOLD_OFF) does not.
    assert "FIG: TRIM" in out and "NVDA: BUY_NOW" in out
    assert "AMD: HOLD_OFF" not in out.split("Action summary")[1]


def test_digest_no_action_case():
    verdicts = [{"symbol": "X", "held": True, "rating": "Buy", "base_action": "HOLD",
                 "action": "HOLD", "escalated": False, "unrealized_pnl_pct": None, "weight_pct": None}]
    out = digest_mod.build_digest("2026-06-15", None, verdicts, None)
    assert "No action required" in out


# ------------------------------- bot parsing ---------------------------------

@pytest.mark.parametrize("text,expected", [
    ("/us NVDA", ("NVDA", "US", "en")),
    ("/hk 0700", ("0700.HK", "HK", "zh")),
    ("/us NVDA zh", ("NVDA", "US", "zh")),
    ("/hk 0700 en", ("0700.HK", "HK", "en")),
    ("/hk 9988.HK", ("9988.HK", "HK", "zh")),
    ("hello", None),
    ("/fr 0700", None),
])
def test_parse_command(text, expected):
    assert bot.parse_command(text) == expected


# ------------------------- positions overlay integration ---------------------

def test_overlay_for_runs(monkeypatch):
    monkeypatch.setattr(positions_mod, "_fx_to_usd", lambda cur: 1.0)
    book = {
        "held": [{"symbol": "FIG", "market": "US", "shares": 100,
                  "cost_basis_per_share": 50.0, "currency": "USD"}],
        "watchlist": {}, "cash": [],
    }
    runs = [
        {"ticker": "FIG", "decision": "Final rating: Sell — deteriorating"},
        {"ticker": "NVDA", "decision": "Rating: Buy"},
    ]
    prices = {"FIG": 60.0, "NVDA": 1000.0}
    verdicts = positions_mod.overlay_for_runs(runs, book, macro_level="LOW", prices=prices)
    by = {v["symbol"]: v for v in verdicts}
    assert by["FIG"]["held"] is True
    assert by["FIG"]["rating"] == "Sell"
    assert by["FIG"]["action"] == "DEFENSIVE"        # held + Sell, calm
    assert by["FIG"]["weight_pct"] == 100.0          # only holding
    assert by["NVDA"]["held"] is False
    assert by["NVDA"]["action"] == "BUY_NOW"


def test_overlay_danger_escalates(monkeypatch):
    monkeypatch.setattr(positions_mod, "_fx_to_usd", lambda cur: 1.0)
    # Two holdings so FIG is NOT concentrated — isolates the danger escalation.
    book = {"held": [
        {"symbol": "FIG", "shares": 1, "cost_basis_per_share": 50.0, "currency": "USD", "market": "US"},
        {"symbol": "BIG", "shares": 1000, "cost_basis_per_share": 1000.0, "currency": "USD", "market": "US"},
    ], "watchlist": {}, "cash": []}
    runs = [{"ticker": "FIG", "decision": "Rating: Hold"}]
    v = positions_mod.overlay_for_runs(runs, book, macro_level="CRITICAL",
                                       prices={"FIG": 50.0, "BIG": 1000.0})[0]
    assert v["weight_pct"] < 1.0   # diluted
    assert v["action"] == "TRIM"   # Hold->WATCH escalated one rung under danger
    assert v["escalated"] is True


# ------------------------------- reflect log ---------------------------------

def test_record_outcomes(tmp_path):
    verdicts = [{"symbol": "FIG", "rating": "Sell"}, {"symbol": "NVDA", "rating": "Buy"}]
    path = tmp_path / "outcomes.csv"
    assert reflect.record_overlay_outcomes(verdicts, "2026-06-15", path) == 2
    # idempotent per (ticker, date)
    assert reflect.record_overlay_outcomes(verdicts, "2026-06-15", path) == 0
    assert path.exists()
