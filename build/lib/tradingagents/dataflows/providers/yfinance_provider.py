"""yfinance provider — PIT safety declarations for existing yfinance code.

This module does NOT reimplement yfinance fetching; it documents which of
the existing ``y_finance.py`` / ``yfinance_news.py`` functions are
point-in-time correct.

PIT-safe (filter strictly by date):
  - get_stock_data        — historical OHLCV, indexed by date
  - get_indicators        — derived from PIT-safe OHLCV
  - get_news              — yfinance filters by pub_date
  - get_global_news       — yfinance filters by pub_date
  - get_balance_sheet     — *with* filing-lag adjustment (see below)
  - get_cashflow          — *with* filing-lag adjustment
  - get_income_statement  — *with* filing-lag adjustment

NOT PIT-safe (returns current snapshot):
  - get_fundamentals          — yfinance ``.info`` is always current
  - get_insider_transactions  — no date filter in the underlying API

The financial statements are marked PIT-safe only after the filing-lag fix
in ``filter_financials_by_date`` lands. Until then they leak by ~45-90 days.
"""

from __future__ import annotations

from . import register
from .base import FilingLag


@register("yfinance")
class YFinanceProvider:
    name = "yfinance"
    # Conservative defaults — yfinance does not expose actual filing dates,
    # so we use these to approximate when a fiscal-period statement was
    # actually public. Tunable via config["data_providers"]["yfinance"]
    # ["filing_lag"] in a later commit.
    filing_lag = FilingLag(annual=90, quarterly=45)

    PIT_SAFE = frozenset({
        "get_stock_data",
        "get_indicators",
        "get_news",
        "get_global_news",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
    })
