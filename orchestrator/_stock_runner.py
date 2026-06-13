"""Runs INSIDE stock_analysier's venv (cwd = stock_analysier project root).

Invoked by orchestrator/adapters/stocks.py as a subprocess so the heavy
stock_analysier dependencies (deepseek client, pydantic schemas, data layer)
stay isolated in that project's own environment. Prints a single JSON object
after a marker line so the parent can parse it reliably.

Usage:  python _stock_runner.py <SYMBOL> <US|HK> [--no-persist]
"""
import json
import sys
from pathlib import Path

# cwd is the stock_analysier project root; make its `src` importable.
sys.path.insert(0, str(Path.cwd()))


def _get(obj, *path, default=None):
    cur = obj
    for p in path:
        cur = getattr(cur, p, None)
        if cur is None:
            return default
    return cur


def main() -> int:
    if len(sys.argv) < 3:
        print("<<<ORCH_JSON>>>")
        print(json.dumps({"ok": False, "error": "usage: SYMBOL MARKET"}))
        return 2

    symbol, market = sys.argv[1], sys.argv[2]
    persist = "--no-persist" not in sys.argv

    from src.pipeline.orchestrator import analyze
    from src.notifier.formatter import format_brief_digest

    result = analyze(symbol, market, persist=persist)
    try:
        text = format_brief_digest([result], push_name="on_demand", market_filter=market)
    except Exception as e:  # formatter is best-effort; never fail the run on it
        text = f"{symbol} ({market}) analyzed; digest formatting failed: {e}"

    out = {
        "symbol": result.symbol,
        "market": result.market,
        "current_price": result.current_price,
        "currency": getattr(result, "currency", None),
        "tactical_label": _get(result, "if_held", "tactical", "label"),
        "if_held_action": _get(result, "if_held", "tactical", "action"),
        "recommendation": _get(result, "if_not_held", "recommendation"),
        "quality_score": _get(result, "fundamental", "quality_score"),
        "margin_of_safety_pct": _get(result, "valuation", "margin_of_safety_pct"),
        "devil_verdict": _get(result, "devil_advocate", "verdict"),
        "telegram_text": text,
        # full AnalysisResult so the orchestrator can render the FULL report
        "full": json.loads(result.model_dump_json()),
    }
    print("<<<ORCH_JSON>>>")
    print(json.dumps(out, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
