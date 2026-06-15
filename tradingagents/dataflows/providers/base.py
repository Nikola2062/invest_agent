"""DataProvider protocol — point-in-time correct data access.

Every read takes an explicit ``as_of`` date. Providers declare which of their
methods are PIT-safe via ``PIT_SAFE``; the composite layer logs a warning
when a non-PIT-safe method is called with an ``as_of`` in the past, so
backtest leakage is visible rather than silent.

The protocol does not replace ``interface.route_to_vendor`` — it sits behind
it. Existing yfinance/alpha_vantage code is wrapped to fit this shape; the
agent-tool surface (``get_stock_data``, ``get_fundamentals``, etc.) is
unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import ClassVar, Optional, Protocol, runtime_checkable


# All data methods are addressed by these literals. Keep in sync with
# interface.py's TOOLS_CATEGORIES / VENDOR_METHODS.
DATA_METHODS = (
    "get_stock_data",
    "get_indicators",
    "get_fundamentals",
    "get_balance_sheet",
    "get_cashflow",
    "get_income_statement",
    "get_news",
    "get_global_news",
    "get_insider_transactions",
)


@dataclass(frozen=True)
class FilingLag:
    """Days to add to fiscal period end to estimate public filing date.

    Used by providers (notably yfinance) that don't expose actual filing
    dates. A conservative default keeps backtests honest at the cost of
    occasionally hiding a statement that was actually filed early.
    """
    annual: int = 90    # 10-K typically filed within 60-90 days
    quarterly: int = 45  # 10-Q typically filed within 40-45 days


@runtime_checkable
class DataProvider(Protocol):
    """All providers implement this surface.

    ``PIT_SAFE`` lists method names that are point-in-time correct for this
    provider — i.e. given ``as_of=D``, the result contains only information
    that was public on or before ``D``. Methods not in this set may return
    forward-looking data (e.g. yfinance ``.info`` returns the current
    snapshot regardless of ``as_of``); the composite warns when those are
    invoked in historical mode.
    """

    name: ClassVar[str]
    PIT_SAFE: ClassVar[frozenset[str]]
    filing_lag: ClassVar[FilingLag]


def is_historical(as_of: Optional[str], today: Optional[date] = None) -> bool:
    """True when ``as_of`` is strictly before today (i.e. backtest mode).

    Centralised so the PIT-safety check has a single source of truth.
    """
    if not as_of:
        return False
    from datetime import datetime
    today = today or date.today()
    try:
        return datetime.strptime(as_of, "%Y-%m-%d").date() < today
    except ValueError:
        return False
