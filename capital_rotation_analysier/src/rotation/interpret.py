"""Rule-based interpretation engine per the project docs §4.4-§4.6.

Takes a SignalSnapshot (dict of signal name -> {score, confidence, ...}) and
produces a list of Claim objects with hedged language, confidence, supporting
evidence, and conflicting evidence.

Anti-hallucination rules (§4.5) enforced here:
1. Hedge verbs only — never assert causality or certainty.
2. Suppress narratives whose confidence is below CONFIDENCE_FLOOR.
3. When score-dispersion is high, conflicting evidence must be cited.
4. Never claim exact money flows; flows are referenced as "consistent with".
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field, asdict
from typing import Callable


# §4.4 confidence bucketing
# Below the floor, narrative is suppressed. Tunable without a code edit via env var.
CONFIDENCE_FLOOR = float(os.environ.get("ROTATION_CONFIDENCE_FLOOR", "0.35"))
CONFIDENCE_LOW = 0.50
CONFIDENCE_MED = 0.70

# §4.3 score-magnitude triggers
STRONG = 50.0
MODERATE = 25.0


def confidence_bucket(c: float | None) -> str:
    if c is None:
        return "unknown"
    if c < CONFIDENCE_FLOOR:
        return "below_floor"
    if c < CONFIDENCE_LOW:
        return "low"
    if c < CONFIDENCE_MED:
        return "medium"
    return "high"


def hedge_for(c: float | None) -> str:
    b = confidence_bucket(c)
    return {
        "high":   "suggests",
        "medium": "appears consistent with",
        "low":    "tentatively consistent with",
        "below_floor": "is inconclusive about",
        "unknown": "cannot determine",
    }[b]


@dataclass
class Claim:
    narrative_id: str
    text: str
    confidence: float
    bucket: str
    supporting: list[str] = field(default_factory=list)
    conflicting: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    # The tickers each narrative predicts will strengthen / weaken.
    # The report layer cross-checks these against today's observed r_w and
    # flags each ticker as consistent (✓), contradicting (✗), or mixed (—).
    implicated_strengthening: list[str] = field(default_factory=list)
    implicated_weakening: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


def _fmt_score(name: str, score: float | None, conf: float | None) -> str:
    if score is None:
        return f"{name}: n/a"
    return f"{name}: {score:+.1f} (conf {conf:.2f})" if conf is not None else f"{name}: {score:+.1f}"


# ============================================================
# Narrative library (§4.6)
# ============================================================
# Each narrative is a (trigger, builder) pair. trigger(signals) -> bool;
# builder(signals) -> Claim. Builders are responsible for filling the slot,
# choosing the hedge verb, and listing supporting/conflicting evidence.

# Per-narrative basket definitions: which tickers each narrative implicates.
# Tags match the signals.py baskets so they stay consistent. The report layer
# cross-checks these against observed r_w to mark consistency / contradiction.
NARRATIVE_BASKETS: dict[str, dict[str, list[str]]] = {
    "risk_on_rotation": {
        "strengthening": ["SPY", "QQQ", "IWM", "EEM", "BTC-USD", "HYG", "XLK", "XLY"],
        "weakening":     ["TLT", "GLD", "UUP", "FXF", "XLP", "XLU"],
    },
    "risk_off_rotation": {
        "strengthening": ["TLT", "GLD", "UUP", "FXF", "XLP", "XLU", "SLV"],
        "weakening":     ["SPY", "QQQ", "IWM", "EEM", "BTC-USD", "ETH-USD", "HYG"],
    },
    "growth_acceleration": {
        "strengthening": ["SMH", "XLI", "IYT", "CPER", "EEM", "XLK", "XLY"],
        "weakening":     ["XLU", "XLP", "TLT"],
    },
    "inflation_rising": {
        "strengthening": ["CPER", "USO", "GLD", "SLV", "XLE"],
        "weakening":     ["TLT", "IEF", "LQD"],
    },
    "inflation_easing": {
        "strengthening": ["TLT", "IEF", "LQD"],
        "weakening":     ["CPER", "USO", "GLD", "SLV"],
    },
    "recession_concern": {
        "strengthening": ["TLT", "IEF", "GLD", "XLU", "XLP", "UUP"],
        "weakening":     ["HYG", "XLY", "IYT", "SMH", "EEM"],
    },
    "liquidity_tightening": {
        "strengthening": ["UUP", "FXF"],
        "weakening":     ["BTC-USD", "ETH-USD", "QQQ", "SMH"],
    },
    "unusual_volume": {
        # Symbol-specific — the report layer will fill from the actual volume
        # anomalies table rather than a static basket.
        "strengthening": [],
        "weakening":     [],
    },
}

def _narr_risk_on_rotation(sig: dict) -> Claim | None:
    roo = sig.get("risk_on_off", {})
    cr = sig.get("capital_rotation", {})
    if (roo.get("score") or 0) < MODERATE:
        return None
    c = roo.get("confidence") or 0
    if c < CONFIDENCE_FLOOR:
        return None
    support, conflict = [], []
    support.append(_fmt_score("Risk-On/Off", roo["score"], c))
    if cr.get("score") is not None and cr["score"] > 0:
        support.append(_fmt_score("Capital Rotation", cr["score"], cr.get("confidence")))
    grw = sig.get("growth", {})
    if (grw.get("score") or 0) > 0:
        support.append(_fmt_score("Growth", grw["score"], grw.get("confidence")))
    rec = sig.get("recession", {})
    if (rec.get("score") or 0) > MODERATE:
        conflict.append(_fmt_score("Recession Concern", rec["score"], rec.get("confidence")))
    inf = sig.get("inflation", {})
    if (inf.get("score") or 0) > STRONG:
        conflict.append(_fmt_score("Inflation", inf["score"], inf.get("confidence")))
    verb = hedge_for(c)
    basket = NARRATIVE_BASKETS["risk_on_rotation"]
    return Claim(
        narrative_id="risk_on_rotation",
        text=f"Risk appetite {verb} a constructive bias; risk assets are favoured over defensives.",
        confidence=c, bucket=confidence_bucket(c),
        supporting=support, conflicting=conflict,
        tags=["rotation", "risk_on"],
        implicated_strengthening=basket["strengthening"],
        implicated_weakening=basket["weakening"],
    )


def _narr_risk_off_rotation(sig: dict) -> Claim | None:
    roo = sig.get("risk_on_off", {})
    if (roo.get("score") or 0) > -MODERATE:
        return None
    c = roo.get("confidence") or 0
    if c < CONFIDENCE_FLOOR:
        return None
    cr = sig.get("capital_rotation", {})
    rec = sig.get("recession", {})
    support, conflict = [], []
    support.append(_fmt_score("Risk-On/Off", roo["score"], c))
    if cr.get("score") is not None and cr["score"] < 0:
        support.append(_fmt_score("Capital Rotation", cr["score"], cr.get("confidence")))
    if (rec.get("score") or 0) > 0:
        support.append(_fmt_score("Recession Concern", rec["score"], rec.get("confidence")))
    grw = sig.get("growth", {})
    if (grw.get("score") or 0) > MODERATE:
        conflict.append(_fmt_score("Growth", grw["score"], grw.get("confidence")))
    verb = hedge_for(c)
    basket = NARRATIVE_BASKETS["risk_off_rotation"]
    return Claim(
        narrative_id="risk_off_rotation",
        text=f"Capital {verb} rotating from risk assets toward defensives such as bonds, gold, and reserve currencies.",
        confidence=c, bucket=confidence_bucket(c),
        supporting=support, conflicting=conflict,
        tags=["rotation", "risk_off"],
        implicated_strengthening=basket["strengthening"],
        implicated_weakening=basket["weakening"],
    )


def _narr_growth_acceleration(sig: dict) -> Claim | None:
    g = sig.get("growth", {})
    if (g.get("score") or 0) < STRONG:
        return None
    c = g.get("confidence") or 0
    if c < CONFIDENCE_FLOOR:
        return None
    support = [_fmt_score("Growth", g["score"], c)]
    cr = sig.get("capital_rotation", {})
    if cr.get("score") is not None and cr["score"] > 0:
        support.append(_fmt_score("Capital Rotation", cr["score"], cr.get("confidence")))
    conflict = []
    rec = sig.get("recession", {})
    if (rec.get("score") or 0) > MODERATE:
        conflict.append(_fmt_score("Recession Concern", rec["score"], rec.get("confidence")))
    basket = NARRATIVE_BASKETS["growth_acceleration"]
    return Claim(
        narrative_id="growth_acceleration",
        text=f"Growth-sensitive assets {hedge_for(c)} improving economic expectations.",
        confidence=c, bucket=confidence_bucket(c),
        supporting=support, conflicting=conflict, tags=["growth"],
        implicated_strengthening=basket["strengthening"],
        implicated_weakening=basket["weakening"],
    )


def _narr_inflation_signal(sig: dict) -> Claim | None:
    inf = sig.get("inflation", {})
    if abs(inf.get("score") or 0) < MODERATE:
        return None
    c = inf.get("confidence") or 0
    if c < CONFIDENCE_FLOOR:
        return None
    direction = "rising" if inf["score"] > 0 else "easing"
    support = [_fmt_score("Inflation", inf["score"], c)]
    conflict = []
    rec = sig.get("recession", {})
    if (rec.get("score") or 0) > MODERATE and direction == "rising":
        conflict.append(_fmt_score("Recession Concern", rec["score"], rec.get("confidence")))
    basket = NARRATIVE_BASKETS[f"inflation_{direction}"]
    return Claim(
        narrative_id=f"inflation_{direction}",
        text=f"Commodity and bond moves {hedge_for(c)} inflation expectations {direction}.",
        confidence=c, bucket=confidence_bucket(c),
        supporting=support, conflicting=conflict, tags=["inflation"],
        implicated_strengthening=basket["strengthening"],
        implicated_weakening=basket["weakening"],
    )


def _narr_recession_concern(sig: dict) -> Claim | None:
    rec = sig.get("recession", {})
    if (rec.get("score") or 0) < STRONG:
        return None
    c = rec.get("confidence") or 0
    if c < CONFIDENCE_FLOOR:
        return None
    support = [_fmt_score("Recession Concern", rec["score"], c)]
    conflict = []
    g = sig.get("growth", {})
    if (g.get("score") or 0) > 0:
        conflict.append(_fmt_score("Growth", g["score"], g.get("confidence")))
    roo = sig.get("risk_on_off", {})
    if (roo.get("score") or 0) > 0:
        conflict.append(_fmt_score("Risk-On/Off", roo["score"], roo.get("confidence")))
    basket = NARRATIVE_BASKETS["recession_concern"]
    return Claim(
        narrative_id="recession_concern",
        text=f"Curve, credit spreads, and cyclical leadership {hedge_for(c)} elevated recession risk.",
        confidence=c, bucket=confidence_bucket(c),
        supporting=support, conflicting=conflict, tags=["recession"],
        implicated_strengthening=basket["strengthening"],
        implicated_weakening=basket["weakening"],
    )


def _narr_liquidity_tight(sig: dict) -> Claim | None:
    liq = sig.get("liquidity", {})
    s = liq.get("score")
    if s is None or s > 35:  # liquidity is 0..100, so low = tight
        return None
    c = liq.get("confidence") or 0
    if c < CONFIDENCE_FLOOR:
        return None
    basket = NARRATIVE_BASKETS["liquidity_tightening"]
    return Claim(
        narrative_id="liquidity_tightening",
        text=f"Dollar strength, vol, and high-beta liquidity tells {hedge_for(c)} tightening liquidity conditions.",
        confidence=c, bucket=confidence_bucket(c),
        supporting=[_fmt_score("Liquidity", s, c)],
        conflicting=[],
        tags=["liquidity"],
        implicated_strengthening=basket["strengthening"],
        implicated_weakening=basket["weakening"],
    )


def _narr_volume_anomaly(sig: dict) -> Claim | None:
    rv = sig.get("relative_volume", {})
    s = rv.get("score")
    if s is None or s < 70:
        return None
    c = rv.get("confidence") or 0
    if c < CONFIDENCE_FLOOR:
        return None
    return Claim(
        narrative_id="unusual_volume",
        text=f"Aggregate trading volume {hedge_for(c)} significantly above its 30-day baseline.",
        confidence=c, bucket=confidence_bucket(c),
        supporting=[_fmt_score("Relative Volume", s, c)],
        conflicting=[],
        tags=["volume"],
    )


def _narr_breadth_thrust(sig: dict) -> Claim | None:
    """Breadth-derived narrative is delegated until breadth metrics are wired through.
    Placeholder kept for the registry; returns None at MVP."""
    return None


# ============================================================
# Cross-cutting checks
# ============================================================

def _detect_conflicting_signals(sig: dict) -> list[str]:
    """Flag pairs of signals that point in contradictory directions."""
    conflicts: list[str] = []
    g = (sig.get("growth", {}).get("score") or 0)
    rec = (sig.get("recession", {}).get("score") or 0)
    roo = (sig.get("risk_on_off", {}).get("score") or 0)
    if g > MODERATE and rec > MODERATE:
        conflicts.append(
            f"Growth ({g:+.1f}) and Recession Concern ({rec:+.1f}) are both elevated — "
            "late-cycle pattern or a regime in transition; resolve with caution."
        )
    if roo > MODERATE and rec > STRONG:
        conflicts.append(
            f"Risk-On ({roo:+.1f}) vs Recession Concern ({rec:+.1f}) disagree — "
            "potentially a relief rally inside a deteriorating macro backdrop."
        )
    return conflicts


NARRATIVES: list[Callable[[dict], Claim | None]] = [
    _narr_risk_on_rotation,
    _narr_risk_off_rotation,
    _narr_growth_acceleration,
    _narr_inflation_signal,
    _narr_recession_concern,
    _narr_liquidity_tight,
    _narr_volume_anomaly,
    _narr_breadth_thrust,
]


def interpret(signals: dict[str, dict]) -> dict:
    """Top-level entry. Returns dict with claims + cross-cutting conflict notes."""
    claims = []
    for fn in NARRATIVES:
        try:
            c = fn(signals)
            if c is not None:
                claims.append(c)
        except Exception as exc:
            claims.append(Claim(
                narrative_id=f"error/{fn.__name__}",
                text=f"Interpreter rule {fn.__name__} failed: {exc}",
                confidence=0.0, bucket="unknown",
                supporting=[], conflicting=[], tags=["error"],
            ))

    cross = _detect_conflicting_signals(signals)

    # Sort claims by confidence descending so the report leads with the strongest.
    claims.sort(key=lambda c: c.confidence, reverse=True)

    return {
        "claims": [c.to_dict() for c in claims],
        "cross_signal_conflicts": cross,
    }
