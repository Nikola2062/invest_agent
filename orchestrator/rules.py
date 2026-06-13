"""D1: the cross-layer overlay.

Takes each stock's OWN verdict (from stock_analysier) plus the market context
(macro risk level from risk_analysis + regime from capital_rotation) and
produces a FINAL action. This is a post-hoc overlay — the projects' own
verdicts are never mutated. Pure functions, no I/O, so it is unit-testable
without any of the three projects present.
"""
from __future__ import annotations

import settings

_RULES = settings.CONFIG["rules"]
_ENTRY_RECOS = {"BUY_NOW", "WAIT_FOR_PRICE"}


def is_danger(macro_risk_level: str | None, regime_label: str | None) -> tuple[bool, list[str]]:
    """Danger if the macro gate is elevated OR the regime is risk-off."""
    reasons = []
    danger = False
    if macro_risk_level and macro_risk_level.upper() in {x.upper() for x in _RULES["danger_macro_levels"]}:
        danger = True
        reasons.append(f"macro risk {macro_risk_level}")
    if regime_label and regime_label in set(_RULES["risk_off_regimes"]):
        danger = True
        reasons.append(f"regime {regime_label}")
    return danger, reasons


def _escalate(label: str | None) -> str | None:
    ladder = _RULES["tactical_ladder"]
    cur = label or ""
    if cur not in ladder:
        return label  # unknown label: leave as-is
    i = ladder.index(cur)
    nxt = ladder[min(i + 1, len(ladder) - 1)]
    return nxt or None


def evaluate(
    *,
    symbol: str,
    held: bool,
    base_tactical: str | None,
    base_reco: str | None,
    macro_risk_level: str | None,
    regime_label: str | None,
) -> dict:
    """Return the final action for one symbol.

    {symbol, held, base_tactical, final_tactical, base_reco, final_reco,
     action, reason, escalated(bool), entry_blocked(bool)}
    """
    danger, why = is_danger(macro_risk_level, regime_label)
    final_tactical = base_tactical
    final_reco = base_reco
    escalated = False
    entry_blocked = False

    if held:
        if danger and _RULES["on_danger"].get("escalate_tactical_one_notch"):
            esc = _escalate(base_tactical)
            if esc != base_tactical:
                final_tactical = esc
                escalated = True
        action = f"tactical: {final_tactical or 'hold / monitor'}"
        if escalated:
            action += f" (escalated from {base_tactical or 'none'})"
    else:
        if danger and _RULES["on_danger"].get("block_new_entries") and base_reco in _ENTRY_RECOS:
            final_reco = "HOLD_OFF"
            entry_blocked = True
            action = f"entry blocked (was {base_reco})"
        else:
            action = f"recommendation: {final_reco or 'n/a'}"

    if danger:
        reason = "; ".join(why)
    else:
        reason = f"calm (macro {macro_risk_level or '?'}, regime {regime_label or '?'})"

    return {
        "symbol": symbol,
        "held": held,
        "base_tactical": base_tactical,
        "final_tactical": final_tactical,
        "base_reco": base_reco,
        "final_reco": final_reco,
        "action": action,
        "reason": reason,
        "escalated": escalated,
        "entry_blocked": entry_blocked,
        "danger": danger,
    }


def needs_action(verdict: dict) -> bool:
    """True if the final verdict implies the user should do something now."""
    if verdict["escalated"] or verdict["entry_blocked"]:
        return True
    if verdict["held"] and (verdict["final_tactical"] or "") not in ("", "YELLOW_WATCH"):
        return True
    if (not verdict["held"]) and verdict["final_reco"] == "BUY_NOW":
        return True
    return False
