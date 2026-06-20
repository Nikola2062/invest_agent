"""Alpha Vantage provider — PIT safety declarations.

Alpha Vantage exposes fiscal report dates *and* often a ``reportedDate``
(filing date) on financial statement endpoints, so it can be made
PIT-correct for fundamentals. The existing ``alpha_vantage_fundamentals.py``
code should be audited to confirm it uses ``reportedDate`` rather than
``fiscalDateEnding`` when filtering — that audit is a follow-up.

Until that audit, fundamentals are conservatively marked NOT PIT-safe
to surface the warning during backtests.
"""

from __future__ import annotations

from . import register
from .base import FilingLag


@register("alpha_vantage")
class AlphaVantageProvider:
    name = "alpha_vantage"
    # Alpha Vantage exposes reportedDate on statements, so once the audit
    # confirms reportedDate is used, the filing-lag becomes 0.
    filing_lag = FilingLag(annual=0, quarterly=0)

    PIT_SAFE = frozenset({
        "get_stock_data",
        "get_indicators",
        "get_news",
        "get_global_news",
    })
