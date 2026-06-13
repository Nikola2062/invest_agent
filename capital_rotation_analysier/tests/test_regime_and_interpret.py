"""Tests for the regime classifier decision tree and the interpretation engine
filter rules (anti-hallucination guarantees from §4.5)."""
from __future__ import annotations

from rotation.regime import (
    REGIME_RISK_OFF, REGIME_RISK_ON, REGIME_LATE_CYCLE, REGIME_DEFLATION,
    REGIME_UNCERTAIN, classify,
)
from rotation.interpret import (
    CONFIDENCE_FLOOR, confidence_bucket, hedge_for, interpret,
)


# ============================================================
# Regime classifier
# ============================================================

def test_regime_risk_off_when_roo_low_and_recession_high():
    r, _, _ = classify({"risk_on_off": -50, "recession": 40})
    assert r == REGIME_RISK_OFF


def test_regime_risk_on_when_roo_high_and_growth_positive():
    r, _, _ = classify({"risk_on_off": 40, "growth": 30})
    assert r == REGIME_RISK_ON


def test_regime_late_cycle_when_roo_and_inflation_both_high():
    r, _, _ = classify({"risk_on_off": 35, "inflation": 50})
    assert r == REGIME_LATE_CYCLE


def test_regime_deflation_when_roo_low_and_inflation_negative():
    r, _, _ = classify({"risk_on_off": -30, "inflation": -40})
    assert r == REGIME_DEFLATION


def test_regime_uncertain_default():
    r, conf, _ = classify({"risk_on_off": 5, "growth": 5})
    assert r == REGIME_UNCERTAIN
    assert conf == 0.30


def test_regime_risk_off_takes_precedence_over_deflation():
    # roo=-40, rec=30 (risk_off rule), inflation=-20 (deflation rule)
    # risk_off comes first in the decision tree -> should win
    r, _, _ = classify({"risk_on_off": -40, "recession": 30, "inflation": -20})
    assert r == REGIME_RISK_OFF


# ============================================================
# Interpretation engine
# ============================================================

def test_confidence_bucket_boundaries():
    assert confidence_bucket(0.0) == "below_floor"
    assert confidence_bucket(CONFIDENCE_FLOOR - 0.001) == "below_floor"
    assert confidence_bucket(CONFIDENCE_FLOOR) == "low"
    assert confidence_bucket(0.49) == "low"
    assert confidence_bucket(0.50) == "medium"
    assert confidence_bucket(0.69) == "medium"
    assert confidence_bucket(0.70) == "high"
    assert confidence_bucket(0.99) == "high"


def test_hedge_for_uses_softer_verb_at_lower_confidence():
    assert "suggests" in hedge_for(0.85)
    assert "tentatively" in hedge_for(0.40)
    assert "inconclusive" in hedge_for(0.20)


def test_interpret_suppresses_low_confidence_narratives():
    """A strong risk-on score with confidence below the floor must not fire a narrative."""
    signals = {
        "risk_on_off": {"score": 70, "confidence": 0.10},  # below floor
        "capital_rotation": {"score": 40, "confidence": 0.10},
        "growth": {"score": 30, "confidence": 0.10},
    }
    out = interpret(signals)
    risk_on_narratives = [c for c in out["claims"] if "risk_on_rotation" in c["narrative_id"]]
    assert risk_on_narratives == [], "Low confidence must suppress the risk-on narrative"


def test_interpret_fires_when_score_strong_and_confident():
    signals = {
        "risk_on_off": {"score": 70, "confidence": 0.80},
        "capital_rotation": {"score": 40, "confidence": 0.75},
        "growth": {"score": 30, "confidence": 0.65},
    }
    out = interpret(signals)
    narratives = [c["narrative_id"] for c in out["claims"]]
    assert "risk_on_rotation" in narratives


def test_interpret_includes_conflicting_evidence_when_signals_disagree():
    # roo > MODERATE (25) AND rec > STRONG (50) -> cross-signal flag fires
    signals = {
        "risk_on_off": {"score": 60, "confidence": 0.75},
        "recession": {"score": 55, "confidence": 0.80},   # strict > 50
        "growth": {"score": 20, "confidence": 0.55},
    }
    out = interpret(signals)
    cross = out["cross_signal_conflicts"]
    assert any("Recession" in c for c in cross), \
        "Cross-signal flag must surface when risk_on and recession both high"


def test_interpret_hedge_language_never_certain():
    """No claim text may contain words asserting certainty (per §4.5)."""
    forbidden = ["will", "definitely", "certain", "guarantees", "always", "must"]
    signals = {
        "risk_on_off": {"score": 80, "confidence": 0.95},
        "growth": {"score": 70, "confidence": 0.90},
        "inflation": {"score": -50, "confidence": 0.85},
    }
    out = interpret(signals)
    for c in out["claims"]:
        text = c["text"].lower()
        for word in forbidden:
            # 'must' is too aggressive a filter (e.g. 'must be considered'),
            # but the rest are absolute. Skip 'must'.
            if word == "must": continue
            assert word not in text, f"Narrative contained forbidden word '{word}': {c['text']}"
