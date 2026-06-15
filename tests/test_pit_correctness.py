"""Point-in-time correctness tests for the Phase 1 data layer.

Covers four classes of leakage that previously made backtests dishonest:

  1. ``filter_financials_by_date`` filtered by fiscal-period end instead of
     filing date, so a 2022-12-31 10-K appeared on a 2023-01-15 query even
     though it was filed in late February.
  2. ``get_insider_transactions`` had no ``curr_date`` parameter at all and
     returned every transaction the vendor knew about, including future ones.
  3. ``get_fundamentals`` (yfinance) called ``.info`` which always returns
     the current snapshot — ratios from today bled into past trades.
  4. The ``route_to_vendor`` boundary did not warn when a non-PIT-safe
     method was invoked with an ``as_of`` in the past.

These tests are unit-level: no network, no real vendor calls.
"""

from __future__ import annotations

import logging
from datetime import datetime
from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.dataflows.providers import (
    is_historical,
    pit_safe_methods,
    warn_if_not_pit_safe,
    _REGISTRY,
)
from tradingagents.dataflows.stockstats_utils import filter_financials_by_date


pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Registry + protocol
# ---------------------------------------------------------------------------

def test_all_expected_providers_registered():
    assert set(_REGISTRY) == {"yfinance", "alpha_vantage", "polygon", "edgar"}


def test_edgar_marks_fundamentals_pit_safe():
    # EDGAR is the only provider that gets fundamentals PIT-correct out of
    # the box — yfinance/alpha_vantage don't. If this changes, audit them.
    assert "get_fundamentals" in pit_safe_methods("edgar")
    assert "get_fundamentals" not in pit_safe_methods("yfinance")


def test_yfinance_insider_transactions_marked_unsafe():
    # The underlying yfinance call has no filing-date filter. Our wrapper
    # adds a date-based filter but cannot recover the actual filing date,
    # so we treat it as not-PIT-safe to keep the warning loud.
    assert "get_insider_transactions" not in pit_safe_methods("yfinance")


# ---------------------------------------------------------------------------
# is_historical
# ---------------------------------------------------------------------------

def test_is_historical_past():
    assert is_historical("2020-01-01") is True


def test_is_historical_future():
    assert is_historical("2099-01-01") is False


def test_is_historical_today():
    today = datetime.today().strftime("%Y-%m-%d")
    assert is_historical(today) is False


def test_is_historical_missing_or_bad_input():
    assert is_historical(None) is False
    assert is_historical("") is False
    assert is_historical("not-a-date") is False


# ---------------------------------------------------------------------------
# filter_financials_by_date — the critical leakage fix
# ---------------------------------------------------------------------------

def _statements(*period_ends):
    return pd.DataFrame({pe: [100] for pe in period_ends}, index=["Revenue"])


def test_filter_drops_future_fiscal_periods():
    df = _statements("2022-12-31", "2023-03-31", "2023-06-30")
    out = filter_financials_by_date(df, "2023-04-15", filing_lag_days=0)
    assert list(out.columns) == ["2022-12-31", "2023-03-31"]


def test_filter_with_quarterly_lag_hides_unfiled_statement():
    """A 2022-12-31 10-K wasn't filed until ~Feb-Mar 2023.

    On a 2023-01-15 query with a 45-day filing-lag, the 2022-12-31 fiscal
    period must NOT appear — it wasn't public yet.
    """
    df = _statements("2022-09-30", "2022-12-31")
    out = filter_financials_by_date(df, "2023-01-15", filing_lag_days=45)
    # Only the Sept quarter is past its filing window.
    assert list(out.columns) == ["2022-09-30"]


def test_filter_with_annual_lag_hides_unfiled_annual():
    df = _statements("2022-12-31", "2023-12-31")
    # 2023-12-31 annual not filed until ~end of March 2024.
    out = filter_financials_by_date(df, "2024-02-15", filing_lag_days=90)
    assert list(out.columns) == ["2022-12-31"]


def test_filter_with_lag_zero_preserves_legacy_behavior():
    df = _statements("2022-12-31", "2023-03-31")
    out = filter_financials_by_date(df, "2023-01-15", filing_lag_days=0)
    assert list(out.columns) == ["2022-12-31"]


def test_filter_empty_passthrough():
    empty = pd.DataFrame()
    out = filter_financials_by_date(empty, "2023-01-15", filing_lag_days=45)
    assert out.empty


def test_filter_no_date_passthrough():
    df = _statements("2022-12-31")
    out = filter_financials_by_date(df, "", filing_lag_days=45)
    assert list(out.columns) == ["2022-12-31"]


# ---------------------------------------------------------------------------
# warn_if_not_pit_safe
# ---------------------------------------------------------------------------

def test_warning_emitted_for_unsafe_historical_call(caplog):
    with caplog.at_level(logging.WARNING):
        warn_if_not_pit_safe("yfinance", "get_fundamentals", "2020-01-01")
    assert any("PIT leakage risk" in r.message for r in caplog.records)


def test_no_warning_for_safe_historical_call(caplog):
    with caplog.at_level(logging.WARNING):
        warn_if_not_pit_safe("yfinance", "get_stock_data", "2020-01-01")
    assert not any("PIT leakage risk" in r.message for r in caplog.records)


def test_no_warning_for_live_call(caplog):
    today = datetime.today().strftime("%Y-%m-%d")
    with caplog.at_level(logging.WARNING):
        warn_if_not_pit_safe("yfinance", "get_fundamentals", today)
    assert not any("PIT leakage risk" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# get_insider_transactions — was the worst offender (no date param at all)
# ---------------------------------------------------------------------------

def test_insider_transactions_filters_by_curr_date():
    """yfinance returns all known insider transactions; the wrapper must
    drop rows after curr_date so backtests don't see future filings."""
    from tradingagents.dataflows import y_finance

    fake = pd.DataFrame({
        "Start Date": pd.to_datetime([
            "2022-06-15", "2023-01-10", "2023-08-22",
        ]),
        "Insider": ["A", "B", "C"],
        "Shares": [100, 200, 300],
    })

    class _Stub:
        insider_transactions = fake

    with patch("tradingagents.dataflows.y_finance.yf.Ticker", return_value=_Stub()):
        out = y_finance.get_insider_transactions("AAPL", curr_date="2023-03-01")

    assert "Filtered to transactions on or before 2023-03-01" in out
    assert "2022-06-15" in out
    assert "2023-01-10" in out
    # The Aug 2023 row must NOT appear in a March 2023 backtest.
    assert "2023-08-22" not in out


def test_insider_transactions_empty_after_filter():
    """When curr_date predates every transaction, return the empty notice."""
    from tradingagents.dataflows import y_finance

    fake = pd.DataFrame({
        "Start Date": pd.to_datetime(["2024-01-01"]),
        "Insider": ["A"],
        "Shares": [100],
    })

    class _Stub:
        insider_transactions = fake

    with patch("tradingagents.dataflows.y_finance.yf.Ticker", return_value=_Stub()):
        out = y_finance.get_insider_transactions("AAPL", curr_date="2020-01-01")

    assert "No insider transactions" in out
    assert "2020-01-01" in out


# ---------------------------------------------------------------------------
# get_fundamentals — must degrade in historical mode
# ---------------------------------------------------------------------------

def test_fundamentals_degrades_in_historical_mode():
    """yfinance `.info` is a current snapshot; in historical mode the
    wrapper must suppress time-varying ratios and warn."""
    from tradingagents.dataflows import y_finance

    snapshot = {
        "longName": "Acme Corp",
        "sector": "Technology",
        "industry": "Software",
        "beta": 1.2,
        # Time-varying fields that MUST be suppressed in historical mode:
        "trailingPE": 25.0,
        "marketCap": 500_000_000,
        "fiftyTwoWeekHigh": 200.0,
    }

    class _Stub:
        info = snapshot

    with patch("tradingagents.dataflows.y_finance.yf.Ticker", return_value=_Stub()):
        out = y_finance.get_fundamentals("ACME", curr_date="2020-01-01")

    assert "PIT WARNING" in out
    assert "Acme Corp" in out          # identity field — OK
    assert "Technology" in out
    assert "PE Ratio" not in out       # time-varying — must be suppressed
    assert "Market Cap" not in out
    assert "52 Week High" not in out


def test_fundamentals_live_mode_unchanged():
    """In live mode (curr_date=today / None) the full snapshot is returned."""
    from tradingagents.dataflows import y_finance

    snapshot = {
        "longName": "Acme",
        "trailingPE": 25.0,
        "marketCap": 500_000_000,
    }

    class _Stub:
        info = snapshot

    with patch("tradingagents.dataflows.y_finance.yf.Ticker", return_value=_Stub()):
        out = y_finance.get_fundamentals("ACME", curr_date=None)

    assert "PIT WARNING" not in out
    assert "PE Ratio" in out
    assert "Market Cap" in out


# ---------------------------------------------------------------------------
# route_to_vendor — extraction of as_of for warning machinery
# ---------------------------------------------------------------------------

def test_as_of_extracted_from_positional_args():
    from tradingagents.dataflows.interface import _extract_as_of

    assert _extract_as_of(
        "get_insider_transactions", ("AAPL", "2023-05-01"), {}
    ) == "2023-05-01"
    assert _extract_as_of(
        "get_balance_sheet", ("AAPL", "quarterly", "2023-05-01"), {}
    ) == "2023-05-01"
    assert _extract_as_of(
        "get_stock_data", ("AAPL", "2023-01-01", "2023-05-01"), {}
    ) == "2023-05-01"


def test_as_of_extracted_from_kwargs():
    from tradingagents.dataflows.interface import _extract_as_of

    assert _extract_as_of(
        "get_insider_transactions", ("AAPL",), {"curr_date": "2023-05-01"}
    ) == "2023-05-01"
