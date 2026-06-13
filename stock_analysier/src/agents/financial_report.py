"""Financial Report agent — deep resolution of the last 2 annual + last 2 quarterly periods.

Reads the deterministic FinancialReport snapshot built by data/financials.py and
asks the reasoner model to walk through each statement: how is revenue moving and
WHY, how are margins evolving, what is happening on the balance sheet, what is
the quality of cash flow, and — crucially — the explicit investment implication.

Output is a FinancialDeepResolution that the dashboard renders alongside the raw
period table.
"""
from __future__ import annotations

import logging
from typing import Optional

from src.agents.base import fmt_dollars
from src.llm.client import chat_json
from src.models.schemas import (
    FinancialDeepResolution,
    FinancialPeriodSet,
    FinancialReport,
)

log = logging.getLogger(__name__)


SYSTEM_PROMPT = """You are a senior equity analyst. You are given the most recent 2 annual and 2 quarterly
financial statements for a single company. Your job is to walk through them as a
human analyst would — connect revenue, margins, balance sheet, and cash flow into
a coherent picture, then state the explicit investment implication.

For each metric you discuss, cite the actual numbers (with units / currency) and
the period-over-period delta in BOTH absolute and percentage terms when relevant.
Do not just describe — interpret. "Revenue grew 12% YoY" is a fact; "Revenue grew
12% YoY but the deceleration from 18% the prior period and the simultaneous gross
margin compression from 71% to 67% suggests pricing pressure in the core segment"
is analysis. We want analysis.

Be specific about WHY a number moved when the line items let you infer it. For
example: if Net Income fell faster than Revenue, look at gross margin, opex
growth, interest expense, or one-time items in the data you have. If Operating
Cash Flow diverges from Net Income, comment on accruals, working capital,
or non-cash charges.

For each of the five core fields below, write 2-4 sentences with real numbers:

  - revenue_trend: shape of revenue (annual + quarterly), acceleration vs.
    deceleration, segment hints if the data permits.
  - margin_trend: gross / operating / net margin trajectory; what's driving them
    (cost of revenue moves, opex moves, leverage of R&D or SG&A).
  - balance_sheet_trend: cash, debt, equity, working capital — is the balance
    sheet strengthening or weakening? Any change in capital structure?
  - cash_flow_quality: how does OCF compare to Net Income? Is FCF growing,
    flat, or declining? What is the company doing with cash (capex, dividends,
    buybacks, debt paydown)?
  - capital_allocation_observed: from the cash-flow lines available — pattern
    of capex intensity, dividends, buybacks, debt issuance/repayment. What
    does it tell you about management priorities?

Then:
  - key_positives: 1-4 bullets, the strongest things in these statements.
  - key_red_flags: 1-4 bullets, the things that should worry an investor.
    Be honest. If everything looks clean, return an empty list — don't invent
    flaws.
  - investment_implication: 3-5 sentences. THIS IS THE PUNCHLINE. Tie the
    above together into what it means for a long-term holder vs. a new buyer
    at today's price. Explicitly mention whether the trajectory supports the
    fundamental thesis, undermines it, or is mixed.
  - summary: 1-2 sentence executive summary.

If data is genuinely missing for a period or metric, say so; do not fabricate
numbers. Missing data IS a finding — call it out under key_red_flags or in the
relevant trend field.
"""


def _format_period_set(p: Optional[FinancialPeriodSet], currency: str) -> str:
    if p is None or not p.periods:
        return "  (no data)"
    header = "  Periods (most-recent first): " + ", ".join(p.periods)
    out_lines = [header]

    def _section(title: str, lines):
        if not lines:
            return
        out_lines.append(f"  {title}:")
        for ln in lines:
            cells = "  |  ".join(fmt_dollars(v) for v in ln.values)
            out_lines.append(f"    {ln.label:<22} {cells}")

    _section("Income Statement",  p.income)
    _section("Balance Sheet",     p.balance)
    _section("Cash Flow",         p.cashflow)
    return "\n".join(out_lines)


def _format_user(
    symbol: str,
    name: Optional[str],
    sector: Optional[str],
    report: FinancialReport,
) -> str:
    return f"""SYMBOL: {symbol}
NAME: {name or '(unknown)'}
SECTOR: {sector or '(unknown)'}
REPORTING CURRENCY: {report.currency}

ANNUAL STATEMENTS:
{_format_period_set(report.annual, report.currency)}

QUARTERLY STATEMENTS:
{_format_period_set(report.quarterly, report.currency)}

FETCH NOTES (any gaps in the source data):
{chr(10).join('  - ' + n for n in report.fetch_notes) if report.fetch_notes else '  (none)'}

Walk through the statements as instructed. Cite real numbers with the {report.currency}
units. Connect the dots between income, balance sheet, and cash flow. End with the
explicit investment implication.
"""


def analyze_financials(
    symbol: str,
    name: Optional[str],
    sector: Optional[str],
    report: FinancialReport,
    use_reasoner: bool = True,
) -> FinancialDeepResolution:
    user_msg = _format_user(symbol, name, sector, report)
    return chat_json(
        system=SYSTEM_PROMPT,
        user=user_msg,
        schema=FinancialDeepResolution,
        reasoner=use_reasoner,
        temperature=0.3,
    )
