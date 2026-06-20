"""SEC EDGAR provider — PIT-correct fundamentals from the source of truth.

EDGAR is the authoritative source for US filings: free, official, and —
critically — every filing has a real ``filed`` date. That lets us close
the leakage hole that yfinance can only approximate with a heuristic
filing-lag.

Endpoints used (all under https://data.sec.gov / https://www.sec.gov):

  /files/company_tickers.json
      Ticker → CIK index. Cached locally; refresh manually if you add a
      new IPO.

  /submissions/CIK{cik}.json
      Filing index for a company (form, filing date, accession number).
      Used to map a fiscal-period statement to its actual public date.

  /api/xbrl/companyfacts/CIK{cik}.json
      All reported XBRL facts for a company across all filings, with
      both ``end`` (fiscal period end) and ``filed`` (public date)
      timestamps on every datapoint. The primary source for PIT-safe
      balance sheet / income statement / cash flow.

Compliance:
  - **User-Agent header is required.** SEC rejects requests without one
    that contains an identifier + email. Set ``EDGAR_USER_AGENT`` env
    var or ``edgar_user_agent`` in config.
  - **Rate limit:** 10 requests/second per SEC fair-use policy. The
    rate limiter here is conservative (8 req/s) to leave headroom.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Optional

import requests

from .config import get_config
from .utils import safe_ticker_component

logger = logging.getLogger(__name__)


_HOST = "https://data.sec.gov"
_PUBLIC_HOST = "https://www.sec.gov"
_RATE_LIMIT_RPS = 8.0  # SEC allows 10; leave headroom
_DEFAULT_TIMEOUT = 30


# --- User-Agent + rate limiter -------------------------------------------


def _user_agent() -> str:
    """Resolve the required User-Agent string. Loud warning if missing.

    SEC explicitly requires an identifiable User-Agent for the data API.
    A bare ``python-requests`` UA gets a 403. We fall back to a generic
    string with a warning rather than crashing, so partial configurations
    surface visibly rather than silently breaking only some endpoints.
    """
    import os
    ua = os.environ.get("EDGAR_USER_AGENT") or get_config().get("edgar_user_agent", "")
    if not ua:
        logger.warning(
            "EDGAR_USER_AGENT not set; using a generic fallback. SEC requires "
            "an identifying header — set EDGAR_USER_AGENT='your-name email@example.com' "
            "or the config key 'edgar_user_agent' before using EDGAR in production."
        )
        ua = "TradingAgents research@example.com"
    return ua


class _RateLimiter:
    """Token-bucket rate limiter (thread-safe).

    Cheaper than ``time.sleep(1 / rps)`` between every call because it
    only sleeps when the bucket is empty — bursty access still goes
    through fast as long as the steady-state stays under the limit.
    """

    def __init__(self, rps: float):
        self._interval = 1.0 / rps
        self._next_slot = 0.0
        self._lock = threading.Lock()

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next_slot - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next_slot = max(now, self._next_slot) + self._interval


_RATE_LIMITER = _RateLimiter(_RATE_LIMIT_RPS)


# --- HTTP wrapper --------------------------------------------------------


def _get_json(url: str) -> dict | list:
    """Rate-limited GET returning parsed JSON. Raises on HTTP errors."""
    _RATE_LIMITER.acquire()
    resp = requests.get(
        url,
        headers={"User-Agent": _user_agent(), "Accept": "application/json"},
        timeout=_DEFAULT_TIMEOUT,
    )
    resp.raise_for_status()
    return resp.json()


# --- CIK lookup (cached) -------------------------------------------------


_CIK_CACHE_LOCK = threading.Lock()
_CIK_CACHE: dict[str, str] | None = None


def _cik_cache_path() -> Path:
    base = get_config().get("data_cache_dir", ".")
    return Path(base) / "edgar_ciks.json"


def _load_cik_index() -> dict[str, str]:
    """Load ticker→CIK mapping, fetching from SEC if not cached.

    The full index is small (~700KB) and rarely changes; cache it
    aggressively. Refresh via ``refresh_cik_index()`` after IPOs.
    """
    global _CIK_CACHE
    with _CIK_CACHE_LOCK:
        if _CIK_CACHE is not None:
            return _CIK_CACHE

        path = _cik_cache_path()
        if path.exists():
            try:
                _CIK_CACHE = json.loads(path.read_text(encoding="utf-8"))
                return _CIK_CACHE
            except (json.JSONDecodeError, OSError) as exc:
                logger.warning("CIK cache unreadable, refetching: %s", exc)

        raw = _get_json(f"{_PUBLIC_HOST}/files/company_tickers.json")
        # SEC ships this as {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "..."}, ...}
        mapping: dict[str, str] = {}
        for entry in raw.values() if isinstance(raw, dict) else []:
            ticker = entry.get("ticker", "").upper()
            cik = entry.get("cik_str")
            if ticker and cik is not None:
                mapping[ticker] = f"{int(cik):010d}"

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(mapping, indent=2, sort_keys=True), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not persist CIK cache to %s: %s", path, exc)

        _CIK_CACHE = mapping
        return mapping


def refresh_cik_index() -> dict[str, str]:
    """Force a re-fetch of the ticker→CIK index. Call after IPOs."""
    global _CIK_CACHE
    with _CIK_CACHE_LOCK:
        _CIK_CACHE = None
    path = _cik_cache_path()
    if path.exists():
        try:
            path.unlink()
        except OSError:
            pass
    return _load_cik_index()


def lookup_cik(ticker: str) -> Optional[str]:
    """Return the 10-digit CIK for ``ticker``, or None if unknown.

    EDGAR uses CIKs (Central Index Keys), not tickers, in every other
    endpoint. Subsidiaries / dual-listed names may not appear here —
    callers should fall back to another provider on None.
    """
    if not ticker:
        return None
    return _load_cik_index().get(ticker.upper())


# --- companyfacts fetching -----------------------------------------------


def _companyfacts_url(cik: str) -> str:
    return f"{_HOST}/api/xbrl/companyfacts/CIK{cik}.json"


def fetch_companyfacts(ticker: str, *, cache: bool = True) -> Optional[dict]:
    """Fetch the full XBRL facts blob for ``ticker``. None when unknown.

    Cached to disk indefinitely since the data is append-only — old
    filings never change. New filings show up after a fresh fetch.
    Use ``cache=False`` to force re-download (e.g. after a known
    restatement).
    """
    cik = lookup_cik(ticker)
    if cik is None:
        logger.warning("No CIK for ticker %s — EDGAR cannot resolve it", ticker)
        return None

    cache_dir = Path(get_config().get("data_cache_dir", ".")) / "edgar" / "companyfacts"
    safe = safe_ticker_component(ticker)
    cache_path = cache_dir / f"{safe}.json"

    if cache and cache_path.exists():
        try:
            return json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("companyfacts cache unreadable for %s: %s", ticker, exc)

    try:
        data = _get_json(_companyfacts_url(cik))
    except requests.HTTPError as exc:
        logger.warning("companyfacts fetch failed for %s (CIK %s): %s", ticker, cik, exc)
        return None

    if cache:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data), encoding="utf-8")
        except OSError as exc:
            logger.warning("Could not cache companyfacts for %s: %s", ticker, exc)

    return data


# --- XBRL → statements ---------------------------------------------------


# Map our display labels to the XBRL concept names EDGAR uses. We pick
# US-GAAP first since virtually every public US filer uses it; IFRS
# filers (rare for US-listed) fall back to a secondary tag where useful.
_BALANCE_SHEET_CONCEPTS: list[tuple[str, str]] = [
    ("Total assets", "Assets"),
    ("Total current assets", "AssetsCurrent"),
    ("Cash and equivalents", "CashAndCashEquivalentsAtCarryingValue"),
    ("Total liabilities", "Liabilities"),
    ("Total current liabilities", "LiabilitiesCurrent"),
    ("Long-term debt", "LongTermDebt"),
    ("Total stockholders' equity", "StockholdersEquity"),
]

# Known limitation: each label maps to a single XBRL concept. Modern
# ASC 606 filers (post-2018 ~Apple, Google, etc.) report revenue under
# ``RevenueFromContractWithCustomerExcludingAssessedTax`` instead of
# ``Revenues``, so the Revenue row for those filers will show pre-2018
# data only. A proper fix is to support label → [concept, fallback]
# priority lists; tracked as a follow-up since it doesn't affect PIT
# correctness — only completeness for re-tagged filers.
_INCOME_STMT_CONCEPTS: list[tuple[str, str]] = [
    ("Revenue", "Revenues"),
    ("Cost of revenue", "CostOfRevenue"),
    ("Gross profit", "GrossProfit"),
    ("Operating income", "OperatingIncomeLoss"),
    ("Net income", "NetIncomeLoss"),
    ("Basic EPS", "EarningsPerShareBasic"),
    ("Diluted EPS", "EarningsPerShareDiluted"),
]

_CASHFLOW_CONCEPTS: list[tuple[str, str]] = [
    ("Operating cash flow", "NetCashProvidedByUsedInOperatingActivities"),
    ("Investing cash flow", "NetCashProvidedByUsedInInvestingActivities"),
    ("Financing cash flow", "NetCashProvidedByUsedInFinancingActivities"),
    ("CapEx", "PaymentsToAcquirePropertyPlantAndEquipment"),
]


# 10-K → annual, 10-Q → quarterly. Amended filings (-A) still count.
_ANNUAL_FORMS = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
_QUARTERLY_FORMS = {"10-Q", "10-Q/A"}


@dataclass
class StatementPoint:
    """One PIT-correct data point: concept + value + period-end + filing-date."""

    concept: str
    label: str
    value: float
    period_end: str          # YYYY-MM-DD, fiscal period end
    filed: str               # YYYY-MM-DD, real public date
    form: str                # e.g. "10-K", "10-Q"
    unit: str                # e.g. "USD", "USD/shares"


def _select_unit(unit_dict: dict) -> Optional[str]:
    """Pick the preferred unit for a fact: USD first, then USD/shares, then anything."""
    if "USD" in unit_dict:
        return "USD"
    if "USD/shares" in unit_dict:
        return "USD/shares"
    if unit_dict:
        return next(iter(unit_dict))
    return None


def _extract_concept(
    facts: dict, concept: str, label: str, *, forms: set[str],
) -> list[StatementPoint]:
    """Extract all points for one concept, filtered to ``forms``."""
    us_gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    concept_data = us_gaap.get(concept)
    if not concept_data:
        return []

    unit = _select_unit(concept_data.get("units") or {})
    if unit is None:
        return []

    out: list[StatementPoint] = []
    for entry in concept_data["units"][unit]:
        if entry.get("form") not in forms:
            continue
        # ``end`` is fiscal period end, ``filed`` is the public date.
        period_end = entry.get("end")
        filed = entry.get("filed")
        if not period_end or not filed:
            continue
        try:
            value = float(entry.get("val"))
        except (TypeError, ValueError):
            continue
        out.append(StatementPoint(
            concept=concept, label=label, value=value,
            period_end=period_end, filed=filed,
            form=entry.get("form", ""), unit=unit,
        ))
    return out


def _filter_by_filing_date(
    points: list[StatementPoint], curr_date: Optional[str],
) -> list[StatementPoint]:
    """Drop points filed after ``curr_date``. Empty curr_date is a passthrough."""
    if not curr_date:
        return points
    return [p for p in points if p.filed <= curr_date]


def _latest_per_period(points: list[StatementPoint]) -> list[StatementPoint]:
    """For each (concept, period_end), keep the latest-filed value.

    Restated filings (10-K/A) replace the original — taking the
    latest-filed entry is the right call because that's what an investor
    *would* have known by ``curr_date``.
    """
    by_key: dict[tuple[str, str], StatementPoint] = {}
    for p in points:
        key = (p.concept, p.period_end)
        existing = by_key.get(key)
        if existing is None or p.filed > existing.filed:
            by_key[key] = p
    return list(by_key.values())


def _statement_as_csv(points: list[StatementPoint], concept_order: list[tuple[str, str]]) -> str:
    """Render a list of points to the same CSV-ish shape the yfinance handlers emit.

    Columns are fiscal-period-end dates (most recent first); rows are
    concept labels. Empty cells where a concept wasn't reported for a
    given period.
    """
    if not points:
        return ""
    # Build {period_end: {label: value}}
    periods = sorted({p.period_end for p in points}, reverse=True)
    by_label_period: dict[str, dict[str, float]] = {}
    for p in points:
        by_label_period.setdefault(p.label, {})[p.period_end] = p.value

    header = "," + ",".join(periods)
    lines = [header]
    for label, _concept in concept_order:
        if label not in by_label_period:
            continue
        cells = [
            f"{by_label_period[label].get(per, '')}" for per in periods
        ]
        lines.append(f"{label}," + ",".join(cells))
    return "\n".join(lines)


# --- Public functions matching the existing tool surface -----------------


def get_balance_sheet(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "annual or quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date YYYY-MM-DD"] = None,
) -> str:
    """PIT-correct balance sheet via EDGAR companyfacts.

    Unlike the yfinance equivalent (filtered by fiscal-period-end with a
    heuristic lag), this filters by the actual SEC filing date — so a
    Dec 31 10-K that was filed in late February correctly disappears
    from any backtest dated before then.
    """
    return _render_statement(
        ticker, curr_date, freq,
        _BALANCE_SHEET_CONCEPTS, label="Balance Sheet",
    )


def get_income_statement(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "annual or quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date YYYY-MM-DD"] = None,
) -> str:
    """PIT-correct income statement via EDGAR companyfacts."""
    return _render_statement(
        ticker, curr_date, freq,
        _INCOME_STMT_CONCEPTS, label="Income Statement",
    )


def get_cashflow(
    ticker: Annotated[str, "ticker symbol"],
    freq: Annotated[str, "annual or quarterly"] = "quarterly",
    curr_date: Annotated[str, "current date YYYY-MM-DD"] = None,
) -> str:
    """PIT-correct cash flow statement via EDGAR companyfacts."""
    return _render_statement(
        ticker, curr_date, freq,
        _CASHFLOW_CONCEPTS, label="Cash Flow",
    )


def get_fundamentals(
    ticker: Annotated[str, "ticker symbol"],
    curr_date: Annotated[str, "current date YYYY-MM-DD"] = None,
) -> str:
    """Snapshot of the most-recently-filed key metrics as of ``curr_date``.

    Unlike yfinance's ``.info`` (which is forever current), this only
    pulls XBRL facts filed on or before ``curr_date`` and returns the
    most-recent period's value for each. The result is a PIT-safe
    fundamentals snapshot — the same shape the yfinance handler emits
    in live mode.
    """
    facts = fetch_companyfacts(ticker)
    if facts is None:
        return f"No EDGAR fundamentals for {ticker} (no CIK match or fetch failed)"

    # Pull from all three statement concept lists; we want the headline
    # numbers regardless of which form they appear on.
    all_concepts = _BALANCE_SHEET_CONCEPTS + _INCOME_STMT_CONCEPTS + _CASHFLOW_CONCEPTS

    points: list[StatementPoint] = []
    for label, concept in all_concepts:
        points.extend(_extract_concept(
            facts, concept, label,
            forms=_ANNUAL_FORMS | _QUARTERLY_FORMS,
        ))
    points = _filter_by_filing_date(points, curr_date)
    points = _latest_per_period(points)

    if not points:
        return f"No PIT-safe fundamentals for {ticker} on or before {curr_date or 'today'}"

    # For each concept, pick the latest period available.
    latest_per_concept: dict[str, StatementPoint] = {}
    for p in points:
        cur = latest_per_concept.get(p.concept)
        if cur is None or p.period_end > cur.period_end:
            latest_per_concept[p.concept] = p

    lines = [f"# EDGAR Fundamentals for {ticker.upper()}"]
    if curr_date:
        lines.append(f"# As of: {curr_date}")
    lines.append("# Source: SEC EDGAR companyfacts (PIT-correct, filtered by filing date)\n")
    for _label, concept in all_concepts:
        p = latest_per_concept.get(concept)
        if p is None:
            continue
        lines.append(
            f"{p.label}: {p.value:,.0f} {p.unit} "
            f"(period {p.period_end}, filed {p.filed}, {p.form})"
        )
    return "\n".join(lines)


def _render_statement(
    ticker: str,
    curr_date: Optional[str],
    freq: str,
    concept_order: list[tuple[str, str]],
    *,
    label: str,
) -> str:
    """Shared body for get_balance_sheet / income_statement / cashflow."""
    facts = fetch_companyfacts(ticker)
    if facts is None:
        return f"No EDGAR data for {ticker} (no CIK match or fetch failed)"

    forms = _ANNUAL_FORMS if freq.lower() == "annual" else _QUARTERLY_FORMS
    points: list[StatementPoint] = []
    for display_label, concept in concept_order:
        points.extend(_extract_concept(facts, concept, display_label, forms=forms))
    points = _filter_by_filing_date(points, curr_date)
    points = _latest_per_period(points)

    if not points:
        suffix = f"on or before {curr_date}" if curr_date else "in any form"
        return f"No EDGAR {label.lower()} entries for {ticker} {suffix}"

    csv_body = _statement_as_csv(points, concept_order)
    header = (
        f"# {label} for {ticker.upper()} ({freq})\n"
        f"# Source: SEC EDGAR companyfacts (PIT-correct, filed-date filter)\n"
    )
    if curr_date:
        header += f"# As of: {curr_date}\n"
    return header + "\n" + csv_body
