"""Strategy abstraction for the backtest harness.

A strategy is anything callable as ``strategy(ticker, date) -> Decision``.
This deliberate looseness keeps three callers symmetric:

  - ``AgentStrategy``: the real TradingAgents LangGraph pipeline. Expensive
    (LLM tokens per call) but the thing under test.
  - ``RandomStrategy``: uniform-random rating draws. The honesty check —
    if your agent strategy doesn't beat random over enough trades, the
    LLM is noise.
  - Inline lambdas / stubs in tests: deterministic decisions, no I/O.

The Decision shape is intentionally minimal — just what portfolio
construction needs (rating + the raw text for traceability). Per-ticker
report paths and full state are written to disk by ``TradingAgentsGraph``
itself; the runner records them via ``state_log_path`` so the tearsheet
can link back to original reports.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol

from tradingagents.agents.utils.rating import RATINGS_5_TIER, parse_rating


@dataclass
class Decision:
    """One (ticker, date) decision recorded by the runner."""

    ticker: str
    trade_date: str            # YYYY-MM-DD
    rating: str                # one of RATINGS_5_TIER
    raw_decision: str = ""     # full Portfolio Manager output, for audit
    state_log_path: str = ""   # path to TradingAgentsGraph's JSON state log
    runtime_seconds: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.rating not in RATINGS_5_TIER:
            raise ValueError(
                f"Decision.rating {self.rating!r} is not one of {RATINGS_5_TIER}"
            )


class Strategy(Protocol):
    """Anything callable as ``strategy(ticker, date) -> Decision``."""

    def __call__(self, ticker: str, trade_date: str) -> Decision: ...


class AgentStrategy:
    """Wraps ``TradingAgentsGraph`` so it satisfies the Strategy protocol.

    Imported lazily so the backtest module is usable without LangGraph
    installed (e.g. when running only the metrics/report layer on a
    previously-recorded JSONL).
    """

    def __init__(self, graph: Any | None = None, **graph_kwargs: Any) -> None:
        if graph is None:
            from tradingagents.graph.trading_graph import TradingAgentsGraph
            graph = TradingAgentsGraph(**graph_kwargs)
        self._graph = graph

    def __call__(self, ticker: str, trade_date: str) -> Decision:
        import time
        t0 = time.monotonic()
        final_state, signal = self._graph.propagate(ticker, trade_date)
        rating = signal if signal in RATINGS_5_TIER else parse_rating(
            final_state.get("final_trade_decision", "")
        )
        state_log_path = ""
        # TradingAgentsGraph writes per-date state logs; recover the path
        # from its in-memory dict so the tearsheet can deep-link.
        log_states = getattr(self._graph, "log_states_dict", {}) or {}
        if str(trade_date) in log_states:
            state_log_path = log_states[str(trade_date)].get("state_log_path", "")
        return Decision(
            ticker=ticker,
            trade_date=str(trade_date),
            rating=rating,
            raw_decision=final_state.get("final_trade_decision", ""),
            state_log_path=state_log_path,
            runtime_seconds=time.monotonic() - t0,
        )


class RandomStrategy:
    """Uniform-random baseline. Seedable for reproducibility.

    The point of this strategy is *not* to make money — it's to give the
    agent strategy something honest to beat. Any LLM stack that doesn't
    materially outperform RandomStrategy over a few hundred trades is
    expensive noise.
    """

    def __init__(self, seed: int | None = 0, ratings: tuple[str, ...] = RATINGS_5_TIER) -> None:
        self._rng = random.Random(seed)
        self._ratings = ratings

    def __call__(self, ticker: str, trade_date: str) -> Decision:
        return Decision(
            ticker=ticker,
            trade_date=str(trade_date),
            rating=self._rng.choice(self._ratings),
            raw_decision="(random baseline)",
        )


def callable_strategy(fn: Callable[[str, str], str]) -> Strategy:
    """Adapt a plain ``(ticker, date) -> rating`` function to a Strategy.

    Useful for tests and quick experiments where you just want to return a
    rating string without bothering with Decision construction.
    """
    def _adapter(ticker: str, trade_date: str) -> Decision:
        return Decision(
            ticker=ticker, trade_date=str(trade_date), rating=fn(ticker, trade_date),
        )
    return _adapter
