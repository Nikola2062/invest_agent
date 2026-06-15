"""Portfolio-level concerns: sizing, risk constraints, sector exposure.

This package sits *after* the agent stack — agents emit ratings per
ticker; the sizer turns a basket of (ticker, rating) decisions plus
market data into target weights with risk constraints applied.

Keeping sizing out of the agents has two benefits:

  1. The LLM prompts stay focused on stock-picking and don't need to
     reason about correlations or vol-targets.
  2. The same sizer can be swapped in for live trading and for backtests,
     so the equity curve reflects the policy that would actually trade.
"""

from .sizing import (
    SizingConfig,
    Sizer,
    equal_weight_sizer,
    risk_aware_sizer,
)
from .position_overlay import (
    OverlayConfig,
    Position,
    TACTICAL_LADDER,
    ENTRY_ACTIONS,
    evaluate,
    is_danger,
    needs_action,
    unrealized_pnl_pct,
    holding_days,
)
from .relative_strength import (
    relative_strength_score,
    rank,
    coarse_regime,
    fetch_prices,
    fetch_and_rank,
    fetch_regime,
    DEFAULT_LOOKBACKS,
)

__all__ = [
    "Sizer",
    "SizingConfig",
    "equal_weight_sizer",
    "risk_aware_sizer",
    # position-aware overlay (Phase 1 / 2.1)
    "OverlayConfig",
    "Position",
    "TACTICAL_LADDER",
    "ENTRY_ACTIONS",
    "evaluate",
    "is_danger",
    "needs_action",
    "unrealized_pnl_pct",
    "holding_days",
    # cross-sectional relative strength + coarse regime (Phase 3)
    "relative_strength_score",
    "rank",
    "coarse_regime",
    "fetch_prices",
    "fetch_and_rank",
    "fetch_regime",
    "DEFAULT_LOOKBACKS",
]
