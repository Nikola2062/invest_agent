"""SEC EDGAR provider — stub.

EDGAR is the authoritative source for US filings and is **free**. The key
property: every filing has a real ``filedAt`` timestamp, so PIT correctness
on fundamentals is achievable without heuristic filing-lag.

To activate this provider:
  1. Implement the methods below using https://www.sec.gov/edgar/sec-api-documentation.
  2. Respect SEC rate limits (10 req/s) and User-Agent requirement.
  3. Register concrete fetch functions in
     ``tradingagents/dataflows/interface.py``'s VENDOR_METHODS table under
     the ``edgar`` key.
  4. Set ``data_vendors.fundamental_data = "edgar"`` in config.

Endpoints to wire up first:
  - /submissions/CIK{cik}.json                         → filing index per company
  - /api/xbrl/companyfacts/CIK{cik}.json               → all reported facts
  - /api/xbrl/companyconcept/CIK{cik}/us-gaap/{tag}.json → single concept
  - /cgi-bin/browse-edgar?action=getcompany&type=4     → Form 4 insider txns

Reference: https://www.sec.gov/edgar/sec-api-documentation
"""

from __future__ import annotations

from . import register
from .base import FilingLag


@register("edgar")
class EdgarProvider:
    name = "edgar"
    # EDGAR exposes real filing dates — no heuristic lag needed.
    filing_lag = FilingLag(annual=0, quarterly=0)

    # EDGAR is the gold standard for US fundamentals + Form 4 insider data.
    PIT_SAFE = frozenset({
        "get_fundamentals",
        "get_balance_sheet",
        "get_cashflow",
        "get_income_statement",
        "get_insider_transactions",
    })
