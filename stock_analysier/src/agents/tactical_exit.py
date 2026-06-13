"""Tactical Exit / Re-entry agent — pure deterministic policy applier.

No LLM involved. Walks the configured risk_policy ladder against the
RiskAssessment and produces a sell+rebuy plan (or no action), with volatility
and tax-awareness adjustments layered on.

Per the investment framework, Ch.6 / Ch.3: this is the discipline layer that prevents
"tactical risk management" from degrading into vibes-driven trading.
"""
from __future__ import annotations

import logging
from datetime import date
from typing import Optional

from src.models.schemas import (
    ActionConfig,
    Holding,
    RiskAssessment,
    RiskPolicy,
    TacticalAction,
    TechnicalAssessment,
    TechnicalConfig,
)

log = logging.getLogger(__name__)


def _prob_at_magnitude(risk: RiskAssessment, magnitude_pct: float) -> float:
    """Linear-interpolated P(drawdown >= magnitude_pct) from the 10/15/20/25 grid."""
    keys = [10, 15, 20, 25]
    probs = [risk.drawdown_probabilities.get(str(k), 0.0) for k in keys]

    if magnitude_pct <= keys[0]:
        return probs[0]
    if magnitude_pct >= keys[-1]:
        return probs[-1]

    for i in range(len(keys) - 1):
        lo, hi = keys[i], keys[i + 1]
        if lo <= magnitude_pct <= hi:
            alpha = (magnitude_pct - lo) / (hi - lo)
            return probs[i] * (1 - alpha) + probs[i + 1] * alpha

    return 0.0


def _apply_vol_adjustment(magnitude_pct: float, risk: RiskAssessment, policy: RiskPolicy) -> float:
    if not policy.volatility_adjustment.enable:
        return magnitude_pct
    rv = risk.realized_vol_annualized_pct
    if rv is None or rv != rv:  # NaN
        return magnitude_pct
    if rv > policy.volatility_adjustment.high_vol_threshold_annualized_pct:
        return magnitude_pct * policy.volatility_adjustment.high_vol_factor
    if rv < policy.volatility_adjustment.low_vol_threshold_annualized_pct:
        return magnitude_pct * policy.volatility_adjustment.low_vol_factor
    return magnitude_pct


def _technical_confirms_bearish(technical: Optional[TechnicalAssessment]) -> bool:
    """Phase 6 boost: returns True iff price-action confirms a sell setup.

    Requires BOTH structure AND volume to agree, on top of the composite vote.
    The composite alone isn't enough — we want explicit price-action confirmation
    so the boost can't fire on, say, a strong RS-driven bearish composite while
    structure is still in an uptrend.
    """
    if technical is None:
        return False
    return (
        technical.composite_signal in ("bearish", "strong_bearish")
        and technical.structure.trend in ("downtrend", "strong_downtrend")
        and technical.volume.institutional_flow == "distribution"
    )


def _trigger_fires(
    action_cfg: ActionConfig,
    risk: RiskAssessment,
    policy: RiskPolicy,
    symbol: Optional[str] = None,
    technical: Optional[TechnicalAssessment] = None,
    tech_cfg: Optional[TechnicalConfig] = None,
) -> bool:
    """Returns True if this risk-policy action level should fire.

    `tech_cfg` is the loaded TechnicalConfig (or None) — passed in from
    `decide_tactical` so we don't re-read the YAML once per action level.
    """
    adjusted_magnitude = _apply_vol_adjustment(
        action_cfg.trigger.drawdown_magnitude_pct, risk, policy
    )
    prob = _prob_at_magnitude(risk, adjusted_magnitude)
    if prob < action_cfg.trigger.probability_min:
        return False

    # Persistence enforcement via audit trail.
    # The signal must have fired in at least N of the last N recent runs (per persistence_days).
    if action_cfg.trigger.persistence_days > 0 and symbol:
        try:
            from src.storage.audit import get_recent_runs  # lazy import to avoid circulars
            window = action_cfg.trigger.persistence_days
            # Pull enough history to evaluate BOTH the persistence window and the cold-start threshold.
            history_limit = max(window, policy.cold_start_min_history_runs)
            recent = get_recent_runs(symbol, limit=history_limit)
        except Exception as e:
            log.warning("Persistence check audit-read failed for %s — allowing trigger: %s", symbol, e)
            return True

        # Cold-start guard: if we don't have enough total history yet, cap the
        # action level that's allowed to first-fire. Higher-level actions must
        # wait for history to accumulate; the action loop in decide_tactical
        # will fall through to a lower (safer) action on the same signal.
        if len(recent) < policy.cold_start_min_history_runs:
            if action_cfg.level > policy.cold_start_max_action_level:
                log.info(
                    "Cold-start cap for %s level=%s: only %d/%d audit runs; capped at level %d.",
                    symbol, action_cfg.label, len(recent),
                    policy.cold_start_min_history_runs, policy.cold_start_max_action_level,
                )
                return False
            log.info(
                "Cold-start first-fire allowed for %s level=%s (under cap %d).",
                symbol, action_cfg.label, policy.cold_start_max_action_level,
            )
            return True

        # Warm history path: need at least (persistence_days - 1) PRIOR runs
        # where this level OR a higher one fired.
        recent_window = recent[:window]
        qualifying = sum(
            1 for r in recent_window
            if r["tactical_level"] is not None and r["tactical_level"] >= action_cfg.level
        )
        required = window - 1  # this run + N-1 historical = N total

        # Phase 6 technical confirmation boost: if structure + volume + composite
        # all agree on a bearish setup, reduce the persistence requirement by 1.
        # Cannot override the cold-start cap above (that check returns earlier).
        if (
            tech_cfg is not None
            and tech_cfg.integration.tactical_persistence_boost
            and _technical_confirms_bearish(technical)
            and required > 1
        ):
            log.info(
                "Tactical Exit %s level=%s: technical confirmation boost "
                "(required %d → %d).",
                symbol, action_cfg.label, required, required - 1,
            )
            required -= 1

        if qualifying < required:
            log.info(
                "Persistence check for %s level=%s: %d/%d prior runs qualifying — fails.",
                symbol, action_cfg.label, qualifying, required,
            )
            return False
    return True


def _days_held(holding: Holding) -> Optional[int]:
    if holding.purchase_date is None:
        return None
    return (date.today() - holding.purchase_date).days


def _apply_tax_guard(
    chosen: ActionConfig,
    holding: Holding,
    policy: RiskPolicy,
    actions_sorted: list[ActionConfig],
) -> tuple[ActionConfig, list[str]]:
    """Returns the (possibly downgraded) action plus human-readable tax notes."""
    notes: list[str] = []
    if not policy.tax_awareness.enable_long_term_holding_check:
        return chosen, notes
    if holding.market != "US":
        return chosen, notes
    held = _days_held(holding)
    if held is None:
        notes.append("Purchase date unknown — cannot evaluate long-term cap gains status.")
        return chosen, notes

    long_term_threshold = policy.tax_awareness.long_term_holding_days_us
    proximity = policy.tax_awareness.long_term_proximity_window_days

    if held < long_term_threshold and (long_term_threshold - held) <= proximity:
        notes.append(
            f"US: position held {held}d, only {long_term_threshold - held}d until long-term capital gains rate. "
            f"Selling now would be taxed at short-term rates."
        )
        if (
            policy.tax_awareness.prefer_trim_over_full_exit_when_close_to_long_term
            and chosen.action == "full_exit"
        ):
            # Downgrade to highest non-exit action (defensive_reduction if available)
            for a in actions_sorted:
                if a.action == "defensive_reduction":
                    notes.append("Downgraded BLACK_EXIT to RED_DEFENSIVE to preserve long-term cap gains window.")
                    return a, notes
    elif held >= long_term_threshold:
        notes.append(f"US: position is long-term ({held}d held). Gains taxed at long-term rate.")

    return chosen, notes


def _load_tech_cfg() -> Optional[TechnicalConfig]:
    """Load TechnicalConfig once per decide_tactical call. Returns None on failure,
    which silently degrades the persistence-boost and rebuy-anchor features."""
    try:
        from src.config.loader import load_technical
        return load_technical()
    except (FileNotFoundError, ImportError) as e:
        log.info("Technical config not available: %s", e)
        return None
    except Exception as e:
        # Unexpected loader failure — surface it, don't silently swallow.
        log.warning("Failed to load technical config (boost + anchoring disabled): %s", e)
        return None


def decide_tactical(
    holding: Holding,
    current_price: float,
    risk: RiskAssessment,
    policy: RiskPolicy,
    technical: Optional[TechnicalAssessment] = None,
) -> TacticalAction:
    """Walks policy.actions from highest level to lowest, returns the first that fires.

    `technical` is optional — when provided, may give a +1 persistence-day boost
    iff structure + volume agree on a bearish setup. Cannot override the
    cold-start action-level cap.
    """
    tech_cfg = _load_tech_cfg()

    # Sort high→low so highest-level trigger wins
    actions_sorted = sorted(policy.actions, key=lambda a: -a.level)

    fired: Optional[ActionConfig] = None
    for a in actions_sorted:
        if _trigger_fires(
            a, risk, policy,
            symbol=holding.symbol, technical=technical, tech_cfg=tech_cfg,
        ):
            fired = a
            break

    if fired is None:
        return TacticalAction(
            level=None,
            label=None,
            action="no_action",
            rationale=(
                "No risk-policy trigger fires. "
                f"Drawdown probabilities: {risk.drawdown_probabilities}. "
                "Continue to hold per long-term thesis."
            ),
        )

    # Tax-awareness guard
    chosen, tax_notes = _apply_tax_guard(fired, holding, policy, actions_sorted)

    # Rebuy band — Phase 6 anchors to price_map.key_support when usable, falls back to %-band.
    rebuy_band_low: Optional[float] = None
    rebuy_band_high: Optional[float] = None
    anchored_to_support = False
    if tech_cfg is not None:
        ks = (
            technical.price_map.key_support
            if technical and technical.price_map and technical.price_map.key_support
            else None
        )
        max_pct = tech_cfg.integration.max_band_anchor_distance_pct / 100.0
        min_pct = tech_cfg.integration.min_band_anchor_distance_pct / 100.0
        if (
            tech_cfg.integration.order_anchor_to_price_map
            and ks is not None
            and current_price * (1 - max_pct) <= ks <= current_price * (1 - min_pct)
        ):
            rebuy_band_high = ks                       # tranche 1 fills right at support
            rebuy_band_low = ks * 0.95                 # tranche 2 sits just below in case support breaks
            anchored_to_support = True

    if not anchored_to_support:
        rebuy_band_low_pct = (chosen.rebuy_at_drawdown_pct or [None, None])[1]
        rebuy_band_high_pct = (chosen.rebuy_at_drawdown_pct or [None, None])[0]
        if rebuy_band_low_pct is not None and rebuy_band_high_pct is not None:
            rebuy_band_low = current_price * (1 - rebuy_band_low_pct / 100.0)
            rebuy_band_high = current_price * (1 - rebuy_band_high_pct / 100.0)

    # Add wash-sale note if applicable
    if holding.market == "US" and current_price < holding.cost_basis_per_share:
        tax_notes.append(
            f"US wash-sale: sale at current price (${current_price:.2f}) vs cost basis "
            f"(${holding.cost_basis_per_share:.2f}) realizes a LOSS. "
            f"Re-buying within {policy.tax_awareness.wash_sale_avoidance_days} days disallows that loss for tax purposes."
        )

    rationale = (
        f"{chosen.label} triggered: "
        f"P(drawdown ≥ {chosen.trigger.drawdown_magnitude_pct:.0f}% vol-adjusted) "
        f"= {_prob_at_magnitude(risk, _apply_vol_adjustment(chosen.trigger.drawdown_magnitude_pct, risk, policy)):.2f} "
        f"≥ threshold {chosen.trigger.probability_min:.2f}. "
        f"{chosen.description}"
    )
    if anchored_to_support and rebuy_band_high is not None:
        rationale += f" Rebuy band anchored to price-map key_support ${rebuy_band_high:.2f}."

    return TacticalAction(
        level=chosen.level,
        label=chosen.label,
        action=chosen.action if chosen.action in (
            "monitor_only", "trim", "defensive_reduction", "full_exit"
        ) else "no_action",
        trim_pct_of_position=chosen.trim_pct_of_position,
        rebuy_band_low=rebuy_band_low,
        rebuy_band_high=rebuy_band_high,
        hedge_recommended=bool(chosen.hedge_remainder),
        rationale=rationale,
        tax_notes=tax_notes,
    )
