"""Unit tests for the D1 overlay and ticker parsing — no projects needed."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import rules  # noqa: E402
import telegram_bot  # noqa: E402


# --- danger detection ------------------------------------------------------
def test_calm_when_low_and_risk_on():
    danger, why = rules.is_danger("LOW", "Risk-On Expansion")
    assert danger is False and why == []


def test_danger_on_critical_macro():
    danger, why = rules.is_danger("CRITICAL", "Risk-On Expansion")
    assert danger is True and any("CRITICAL" in w for w in why)


def test_danger_on_risk_off_regime():
    danger, why = rules.is_danger("LOW", "Risk-Off Defensive")
    assert danger is True and any("Risk-Off" in w for w in why)


# --- held escalation -------------------------------------------------------
def test_held_escalates_one_notch_under_danger():
    v = rules.evaluate(symbol="FIG", held=True, base_tactical="YELLOW_WATCH",
                       base_reco=None, macro_risk_level="CRITICAL",
                       regime_label="Risk-On Expansion")
    assert v["escalated"] is True
    assert v["final_tactical"] == "ORANGE_TRIM"


def test_held_no_escalation_when_calm():
    v = rules.evaluate(symbol="FIG", held=True, base_tactical="YELLOW_WATCH",
                       base_reco=None, macro_risk_level="LOW",
                       regime_label="Risk-On Expansion")
    assert v["escalated"] is False
    assert v["final_tactical"] == "YELLOW_WATCH"


def test_black_exit_does_not_overflow_ladder():
    v = rules.evaluate(symbol="X", held=True, base_tactical="BLACK_EXIT",
                       base_reco=None, macro_risk_level="CRITICAL", regime_label=None)
    assert v["final_tactical"] == "BLACK_EXIT"


# --- entry blocking --------------------------------------------------------
def test_entry_blocked_under_danger():
    v = rules.evaluate(symbol="NVDA", held=False, base_tactical=None,
                       base_reco="BUY_NOW", macro_risk_level="HIGH", regime_label=None)
    assert v["entry_blocked"] is True
    assert v["final_reco"] == "HOLD_OFF"
    assert rules.needs_action(v) is True


def test_entry_allowed_when_calm():
    v = rules.evaluate(symbol="NVDA", held=False, base_tactical=None,
                       base_reco="BUY_NOW", macro_risk_level="LOW",
                       regime_label="Risk-On Expansion")
    assert v["entry_blocked"] is False
    assert v["final_reco"] == "BUY_NOW"


# --- command parsing (explicit market + language) -------------------------
def test_parse_us_command_defaults_english():
    assert telegram_bot.parse_command("/us NVDA") == ("NVDA", "US", "en")
    assert telegram_bot.parse_command("us msft") == ("MSFT", "US", "en")
    assert telegram_bot.parse_command("/us BRK.B") == ("BRK.B", "US", "en")


def test_parse_hk_command_defaults_chinese_and_adds_suffix():
    # /hk is the Chinese indicator → lang defaults to zh.
    assert telegram_bot.parse_command("/hk 0700") == ("0700.HK", "HK", "zh")
    assert telegram_bot.parse_command("hk 9988.hk") == ("9988.HK", "HK", "zh")


def test_parse_language_override():
    assert telegram_bot.parse_command("/us NVDA zh") == ("NVDA", "US", "zh")
    assert telegram_bot.parse_command("/us NVDA 中文") == ("NVDA", "US", "zh")
    assert telegram_bot.parse_command("/hk 0700 en") == ("0700.HK", "HK", "en")


def test_parse_requires_explicit_market():
    # A bare ticker without /us or /hk is rejected — market must be explicit.
    assert telegram_bot.parse_command("NVDA") is None
    assert telegram_bot.parse_command("0700.HK") is None
    assert telegram_bot.parse_command("hello world") is None
    assert telegram_bot.parse_command("") is None


# --- holiday-aware scheduling ----------------------------------------------
def _load_main():
    import importlib.util
    root = Path(__file__).resolve().parents[2]   # repo root holds main.py
    spec = importlib.util.spec_from_file_location("orch_main", root / "main.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_trading_day_skips_weekend_and_holiday():
    import pytest
    mcal = pytest.importorskip("pandas_market_calendars")
    main = _load_main()
    nyse = mcal.get_calendar("XNYS")
    hkex = mcal.get_calendar("XHKG")
    assert main._trading_day(nyse, "2026-06-12") is True     # Friday — open
    assert main._trading_day(nyse, "2026-06-13") is False    # Saturday — closed
    assert main._trading_day(nyse, "2026-07-03") is False    # US Independence Day (observed)
    assert main._trading_day(hkex, "2026-10-01") is False    # HK National Day holiday
