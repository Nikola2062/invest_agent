"""Walk-forward backtest harness for the TradingAgents framework.

Decouples the *decision producer* (LLM agent graph, random baseline, or any
custom callable) from the *portfolio + metrics* layer so:

  - Real LLM runs and cheap baselines share one harness
  - Tests exercise the full pipeline without API keys or network
  - Phase 3's sizing layer plugs into ``portfolio.py`` without touching the
    runner or strategy interface

Public entry points:

  >>> from tradingagents.backtest import Strategy, Decision, walk_forward
  >>> from tradingagents.backtest.metrics import summarise
  >>> from tradingagents.backtest.report import render_markdown
"""

from .strategy import AgentStrategy, Decision, RandomStrategy, Strategy

__all__ = [
    "AgentStrategy",
    "Decision",
    "RandomStrategy",
    "Strategy",
]
