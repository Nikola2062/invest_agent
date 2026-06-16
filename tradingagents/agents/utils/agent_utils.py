from langchain_core.messages import HumanMessage, RemoveMessage

# Import tools from separate utility files
from tradingagents.agents.utils.core_stock_tools import (
    get_stock_data
)
from tradingagents.agents.utils.technical_indicators_tools import (
    get_indicators
)
from tradingagents.agents.utils.fundamental_data_tools import (
    get_fundamentals,
    get_balance_sheet,
    get_cashflow,
    get_income_statement
)
from tradingagents.agents.utils.news_data_tools import (
    get_news,
    get_insider_transactions,
    get_global_news
)


def get_language_instruction() -> str:
    """Return a prompt instruction for the configured output language.

    Returns empty string when English (default), so no extra tokens are used.
    Applied to every agent whose output reaches the saved report —
    analysts, researchers, debaters, research manager, trader, and
    portfolio manager — so a non-English run produces a fully localized
    report rather than a mix of languages.
    """
    from tradingagents.dataflows.config import get_config
    lang = get_config().get("output_language", "English")
    if lang.strip().lower() == "english":
        return ""
    return f" Write your entire response in {lang}."


def _compress_report(text: str, max_chars: int) -> str:
    """Length-bound an analyst report for replay inside debate prompts.

    Keeps the opening (thesis / overview) and the closing (which is where
    these analysts put their summary table and recommendation), dropping the
    middle with a marker. Snaps to line boundaries so nothing is cut mid-line.
    Deterministic and free — no extra LLM call. The full report is always kept
    verbatim in state and the HTML report; this only shapes the debate context.
    """
    text = (text or "").strip()
    if len(text) <= max_chars:
        return text
    head_budget = max_chars * 3 // 5
    tail_budget = max_chars - head_budget
    # Snap to line boundaries to avoid mid-line cuts, but only when the snap
    # keeps most of the budget — otherwise a long newline-free paragraph would
    # collapse the slice to almost nothing.
    head = text[:head_budget]
    nl = head.rfind("\n")
    if nl >= head_budget // 2:
        head = head[:nl]
    tail = text[-tail_budget:]
    nl = tail.find("\n")
    if nl != -1 and nl <= tail_budget // 2:
        tail = tail[nl + 1 :]
    return (
        head.rstrip()
        + "\n\n…[report condensed for debate — full version in the HTML report]…\n\n"
        + tail.strip()
    )


def get_debate_reports(state) -> tuple[str, str, str, str]:
    """Return (market, sentiment, news, fundamentals) reports for debate prompts.

    The five debate voices (bull, bear, aggressive, neutral, conservative)
    re-ingest all four analyst reports on every turn, so the full text is the
    single biggest repeated token cost in a run. ``debate_report_mode``
    controls how they're served:

      - ``"compact"`` (default): each report is condensed to
        ``debate_report_max_chars`` — large input-token saving, minimal
        analytical loss since headings/summary/recommendation are preserved.
      - ``"full"``: legacy behavior — entire reports in every debate prompt.

    Either way the untouched reports remain in state for the HTML report.
    """
    from tradingagents.dataflows.config import get_config

    reports = (
        state.get("market_report", ""),
        state.get("sentiment_report", ""),
        state.get("news_report", ""),
        state.get("fundamentals_report", ""),
    )
    cfg = get_config()
    if cfg.get("debate_report_mode", "compact") == "full":
        return reports
    budget = int(cfg.get("debate_report_max_chars", 1800))
    return tuple(_compress_report(r, budget) for r in reports)  # type: ignore[return-value]


def build_instrument_context(ticker: str, asset_type: str = "stock") -> str:
    """Describe the exact instrument so agents preserve exchange-qualified tickers."""
    instrument_label = "asset" if asset_type == "crypto" else "instrument"
    extra_hint = (
        " Treat it as a crypto asset rather than a company, and do not assume company fundamentals are available."
        if asset_type == "crypto"
        else ""
    )
    return (
        f"The {instrument_label} to analyze is `{ticker}`. "
        "Use this exact ticker in every tool call, report, and recommendation, "
        "preserving any exchange suffix (e.g. `.TO`, `.L`, `.HK`, `.T`, `-USD`)."
        + extra_hint
    )

def create_msg_delete():
    def delete_messages(state):
        """Clear messages and add placeholder for Anthropic compatibility"""
        messages = state["messages"]

        # Remove all messages
        removal_operations = [RemoveMessage(id=m.id) for m in messages]

        # Add a minimal placeholder message
        placeholder = HumanMessage(content="Continue")

        return {"messages": removal_operations + [placeholder]}

    return delete_messages


        
