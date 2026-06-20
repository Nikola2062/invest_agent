"""Position-aware overlay: turn a per-name 5-tier rating into a *holder's* action.

TradingAgents emits a generic rating (Buy…Sell) that ignores whether you own the
name, at what cost, or how concentrated the book already is — 90% of real investing
is "I already own this, now what?". This **pure, post-hoc** overlay maps the rating
onto a holder's tactical ladder using cost basis, unrealized P&L, holding period and
concentration, then (Phase 2.1) escalates under market danger. It never mutates the
agent's own verdict.

Modelled on ``investor_agent/orchestrator/rules.py`` (pure, unit-tested). Ported as
*logic* — not code — from stock_analysier's tactical_exit / cost_basis / portfolio_fit,
whose machinery is bound to a much heavier risk-policy schema.

No I/O, no LLM: every input is a value, so the whole module is unit-testable offline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Optional

from tradingagents.agents.utils.rating import RATINGS_5_TIER

# Held-position tactical ladder, low -> high severity. "" / HOLD == let it run.
TACTICAL_LADDER = ("HOLD", "WATCH", "TRIM", "DEFENSIVE", "EXIT")
# New-entry recommendations for watchlist (non-held) names.
ENTRY_ACTIONS = ("BUY_NOW", "WAIT_FOR_PRICE", "HOLD_OFF", "AVOID")
_ENTRY_OPENERS = {"BUY_NOW", "WAIT_FOR_PRICE"}  # downgraded under danger

# Rating -> base action. Held: a Buy/Overweight means "let the winner run" (HOLD),
# bearish ratings climb the trim/exit ladder. Entry: bullish ratings open, the rest
# stand aside.
_HELD_BASE = {
    "Buy": "HOLD",
    "Overweight": "HOLD",
    "Hold": "WATCH",
    "Underweight": "TRIM",
    "Sell": "DEFENSIVE",
}
_ENTRY_BASE = {
    "Buy": "BUY_NOW",
    "Overweight": "WAIT_FOR_PRICE",
    "Hold": "HOLD_OFF",
    "Underweight": "AVOID",
    "Sell": "AVOID",
}


@dataclass
class OverlayConfig:
    """Thresholds for the overlay. Defaults mirror orchestrator/config.yaml ∪ a holder's
    common-sense concentration/profit rules."""
    danger_macro_levels: tuple = ("HIGH", "CRITICAL")
    risk_off_regimes: tuple = (
        "Risk-Off Defensive", "Deflationary Shock", "Regime Uncertain",
    )
    concentration_trim_pct: float = 25.0   # > this and not bearish -> trim on strength
    concentration_hard_pct: float = 40.0   # > this -> trim regardless of rating
    big_gain_pct: float = 100.0            # large unrealized gain hardens a bearish call
    stop_loss_pct: float = -25.0           # underwater past this + Sell -> EXIT
    long_term_days: int = 365              # US long-term capital-gains threshold (tax note)


@dataclass
class Position:
    symbol: str
    rating: str                            # one of RATINGS_5_TIER
    held: bool
    shares: float = 0.0
    cost_basis: Optional[float] = None     # per share
    current_price: Optional[float] = None
    weight_pct: Optional[float] = None     # this position as % of total book value
    purchase_date: Optional[str] = None    # YYYY-MM-DD
    market: str = "US"

    def __post_init__(self):
        if self.rating not in RATINGS_5_TIER:
            raise ValueError(f"rating {self.rating!r} not in {RATINGS_5_TIER}")


def unrealized_pnl_pct(pos: Position) -> Optional[float]:
    if pos.cost_basis and pos.current_price and pos.cost_basis > 0:
        return (pos.current_price - pos.cost_basis) / pos.cost_basis * 100.0
    return None


def holding_days(pos: Position, today: Optional[date] = None) -> Optional[int]:
    if not pos.purchase_date:
        return None
    try:
        d = datetime.strptime(pos.purchase_date, "%Y-%m-%d").date()
    except (ValueError, TypeError):
        return None
    return (((today or date.today()) - d)).days


def _idx(action: str) -> int:
    return TACTICAL_LADDER.index(action) if action in TACTICAL_LADDER else 0


def _escalate(action: str, rungs: int = 1) -> str:
    return TACTICAL_LADDER[min(_idx(action) + rungs, len(TACTICAL_LADDER) - 1)]


def is_danger(macro_level: Optional[str], regime: Optional[str],
              cfg: OverlayConfig) -> tuple[bool, list[str]]:
    """Danger if the macro gate is elevated OR the regime is risk-off (D1 rule)."""
    reasons, danger = [], False
    if macro_level and macro_level.upper() in {x.upper() for x in cfg.danger_macro_levels}:
        danger = True
        reasons.append(f"macro risk {macro_level}")
    if regime and regime in set(cfg.risk_off_regimes):
        danger = True
        reasons.append(f"regime {regime}")
    return danger, reasons


def evaluate(pos: Position, *, macro_level: Optional[str] = None,
             regime: Optional[str] = None, cfg: Optional[OverlayConfig] = None,
             today: Optional[date] = None) -> dict:
    """Return the holder action for one position. Pure; never mutates the rating."""
    cfg = cfg or OverlayConfig()
    danger, why = is_danger(macro_level, regime, cfg)
    pnl = unrealized_pnl_pct(pos)
    held_days = holding_days(pos, today)
    notes: list[str] = []

    if pos.held:
        action = _HELD_BASE[pos.rating]
        base = action

        # Concentration: trim on strength when the book is over-weighted in one name.
        if pos.weight_pct is not None:
            if pos.weight_pct > cfg.concentration_hard_pct:
                action = max(action, "TRIM", key=_idx)   # at least trim, regardless of rating
                notes.append(f"concentration {pos.weight_pct:.0f}% > {cfg.concentration_hard_pct:.0f}% hard cap")
            elif pos.weight_pct > cfg.concentration_trim_pct and pos.rating not in ("Buy", "Overweight"):
                action = max(action, "TRIM", key=_idx)
                notes.append(f"concentration {pos.weight_pct:.0f}% > {cfg.concentration_trim_pct:.0f}%")

        # Profit-taking: a large unrealized gain hardens an already-bearish call.
        if pnl is not None and pnl >= cfg.big_gain_pct and pos.rating in ("Underweight", "Sell"):
            action = _escalate(action)
            notes.append(f"lock gains (+{pnl:.0f}%) on bearish call")

        # Stop discipline: deeply underwater + Sell -> exit, don't average a loser.
        if pnl is not None and pnl <= cfg.stop_loss_pct and pos.rating == "Sell":
            action = "EXIT"
            notes.append(f"stop hit ({pnl:.0f}%)")

        # Danger overlay (Phase 2.1): escalate one rung under market danger.
        escalated = False
        if danger:
            esc = _escalate(action)
            if esc != action:
                action = esc
                escalated = True

        # Tax note (informational only — does not change the action).
        if held_days is not None and pos.market == "US" and held_days < cfg.long_term_days \
                and _idx(action) >= _idx("TRIM"):
            notes.append(f"short-term ({held_days}d < {cfg.long_term_days}d) — gains taxed as ordinary income")

        return {
            "symbol": pos.symbol, "held": True, "rating": pos.rating,
            "base_action": base, "action": action,
            "escalated": escalated, "danger": danger,
            "unrealized_pnl_pct": pnl, "weight_pct": pos.weight_pct,
            "reason": "; ".join(why) if danger else "calm",
            "notes": notes,
        }

    # Not held (watchlist) -> entry recommendation.
    reco = _ENTRY_BASE[pos.rating]
    base = reco
    entry_blocked = False
    if danger and reco in _ENTRY_OPENERS:
        reco = "HOLD_OFF"
        entry_blocked = True
        notes.append("entry blocked under market danger")
    return {
        "symbol": pos.symbol, "held": False, "rating": pos.rating,
        "base_action": base, "action": reco,
        "entry_blocked": entry_blocked, "danger": danger,
        "reason": "; ".join(why) if danger else "calm",
        "notes": notes,
    }


def needs_action(verdict: dict) -> bool:
    """True if the verdict implies the holder should do something now (for the digest)."""
    if verdict["held"]:
        return verdict["action"] not in ("HOLD", "WATCH") or verdict.get("escalated", False)
    return verdict["action"] == "BUY_NOW" or verdict.get("entry_blocked", False)
