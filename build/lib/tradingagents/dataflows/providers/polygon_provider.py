"""Polygon.io provider — stub.

Polygon offers PIT-correct OHLCV (splits/dividends handled), timestamped
news, and corporate-action endpoints. Recommended as the US prices/news
foundation. Sign up at https://polygon.io; set POLYGON_API_KEY.

To activate this provider:
  1. Implement the methods below using https://polygon.io/docs.
  2. Register the concrete fetch functions in
     ``tradingagents/dataflows/interface.py``'s VENDOR_METHODS table under
     the ``polygon`` key.
  3. Set ``data_vendors.core_stock_apis = "polygon"`` (or whichever
     category) in config.

Endpoints to wire up first:
  - /v2/aggs/ticker/{ticker}/range/...  → get_stock_data
  - /v2/reference/news                  → get_news / get_global_news
  - /vX/reference/tickers/{ticker}      → get_fundamentals (snapshot)
  - /v3/reference/insiders              → get_insider_transactions
"""

from __future__ import annotations

from . import register
from .base import FilingLag


@register("polygon")
class PolygonProvider:
    name = "polygon"
    # Polygon doesn't expose statements directly — fundamentals are sourced
    # from a separate provider (EDGAR/FMP). Filing-lag is moot here.
    filing_lag = FilingLag(annual=0, quarterly=0)

    # Methods Polygon natively supports PIT-correct (once implemented).
    PIT_SAFE = frozenset({
        "get_stock_data",
        "get_indicators",
        "get_news",
        "get_global_news",
        "get_insider_transactions",
    })
