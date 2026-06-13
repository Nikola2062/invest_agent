"""Tests for the LLM-driven Section 12. We mock the HTTP layer so these tests
run offline and deterministically."""
from __future__ import annotations

import json
from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from rotation import llm_interpret as L


def _good_signals():
    return {
        "relative_strength":  {"score": -90.5, "confidence": 1.0},
        "relative_volume":    {"score": +63.1, "confidence": 0.81},
        "capital_rotation":   {"score":  +2.3, "confidence": 0.87},
        "risk_on_off":        {"score": -68.5, "confidence": 0.25},
        "inflation":          {"score": -25.4, "confidence": 0.80},
        "growth":             {"score":  -9.6, "confidence": 0.60},
        "recession":          {"score": +29.3, "confidence": 0.80},
        "liquidity":          {"score": +36.3, "confidence": 0.60},
    }


def _good_metrics():
    return pd.DataFrame([
        {"symbol": "SPY", "r_d": -0.02, "r_w": -0.03, "vz": 1.5},
        {"symbol": "TLT", "r_d":  0.005, "r_w":  0.018, "vz": 0.5},
    ])


def _mock_deepseek_response(payload: dict):
    """Build a fake urllib.request.urlopen() resp() context manager."""
    raw = json.dumps({
        "model": "deepseek-test",
        "choices": [{"message": {"content": json.dumps(payload)}}],
        "usage": {"prompt_tokens": 1234, "completion_tokens": 567},
    }).encode()
    cm = MagicMock()
    cm.__enter__ = MagicMock(return_value=MagicMock(read=MagicMock(return_value=raw)))
    cm.__exit__ = MagicMock(return_value=False)
    return cm


# ----------------------------------------------------------------
# Validation + sanitization
# ----------------------------------------------------------------

UNIVERSE = ["SPY", "QQQ", "TLT", "GLD", "BTC-USD"]


def test_validate_accepts_valid_claim():
    payload = {
        "claims": [{
            "narrative_id": "risk_off",
            "text": "Capital appears consistent with rotating toward defensives.",
            "confidence": 0.7,
            "supporting": ["roo -68"],
            "conflicting": ["growth -10"],
            "implicated_strengthening": ["TLT", "GLD"],
            "implicated_weakening": ["SPY"],
            "reasoning": "Why",
        }],
        "cross_signal_conflicts": []
    }
    out = L._validate_and_sanitize(payload, UNIVERSE)
    assert out is not None
    assert len(out["claims"]) == 1
    assert out["claims"][0]["narrative_id"] == "risk_off"


def test_validate_drops_below_floor_confidence():
    payload = {
        "claims": [{
            "narrative_id": "weak",
            "text": "Markets appear consistent with mild risk-on.",
            "confidence": 0.20,   # below CONFIDENCE_FLOOR (0.35)
            "supporting": [], "conflicting": [],
            "implicated_strengthening": [], "implicated_weakening": [],
        }],
        "cross_signal_conflicts": []
    }
    out = L._validate_and_sanitize(payload, UNIVERSE)
    assert out["claims"] == []  # filtered out


def test_validate_drops_hallucinated_tickers():
    payload = {
        "claims": [{
            "narrative_id": "rot",
            "text": "Capital appears consistent with rotation.",
            "confidence": 0.6,
            "supporting": [], "conflicting": [],
            "implicated_strengthening": ["TLT", "FAKEFAKE", "GLD"],
            "implicated_weakening": ["NONEXISTENT"],
        }],
        "cross_signal_conflicts": []
    }
    out = L._validate_and_sanitize(payload, UNIVERSE)
    assert out["claims"][0]["implicated_strengthening"] == ["TLT", "GLD"]
    assert out["claims"][0]["implicated_weakening"] == []


def test_validate_rejects_claims_without_hedge_verb():
    payload = {
        "claims": [{
            "narrative_id": "absolute",
            "text": "Markets will crash tomorrow.",  # no hedge verb + 'will'
            "confidence": 0.8,
            "supporting": [], "conflicting": [],
            "implicated_strengthening": [], "implicated_weakening": [],
        }],
        "cross_signal_conflicts": []
    }
    out = L._validate_and_sanitize(payload, UNIVERSE)
    assert out["claims"] == []


def test_validate_rejects_certainty_words():
    payload = {
        "claims": [{
            "narrative_id": "definite",
            "text": "Data definitely suggests bonds are rallying.",  # 'definitely' forbidden
            "confidence": 0.8,
            "supporting": [], "conflicting": [],
            "implicated_strengthening": ["TLT"], "implicated_weakening": [],
        }],
        "cross_signal_conflicts": []
    }
    out = L._validate_and_sanitize(payload, UNIVERSE)
    assert out["claims"] == []


def test_validate_handles_malformed_payload():
    assert L._validate_and_sanitize("not a dict", UNIVERSE) is None
    assert L._validate_and_sanitize({"claims": "not a list"}, UNIVERSE) is None


def test_validate_drops_claims_with_missing_keys():
    payload = {
        "claims": [{
            "narrative_id": "incomplete",
            "text": "appears consistent with X.",
            "confidence": 0.6,
            # missing supporting, conflicting, implicated_*
        }],
        "cross_signal_conflicts": []
    }
    out = L._validate_and_sanitize(payload, UNIVERSE)
    assert out["claims"] == []


# ----------------------------------------------------------------
# End-to-end with mocked DeepSeek HTTP
# ----------------------------------------------------------------

def test_llm_interpret_returns_none_without_key(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    out = L.llm_interpret(asof="2026-01-01", signals=_good_signals(),
                          metrics=_good_metrics())
    assert out is None


def test_llm_interpret_happy_path(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    payload = {
        "claims": [{
            "narrative_id": "risk_off_rotation",
            "text": "Capital appears consistent with rotating toward defensives.",
            "confidence": 0.65,
            "supporting": ["Risk-On/Off -68.5"],
            "conflicting": ["Capital Rotation +2.3 muddies the rotation read"],
            "implicated_strengthening": ["TLT", "GLD"],
            "implicated_weakening": ["SPY", "QQQ"],
            "reasoning": "RoO deeply negative, recession concern elevated",
        }],
        "cross_signal_conflicts": ["Capital Rotation +2.3 contradicts Risk-On/Off -68.5"]
    }
    with patch.object(L.urllib.request, "urlopen",
                       return_value=_mock_deepseek_response(payload)):
        out = L.llm_interpret(asof="2026-01-01", signals=_good_signals(),
                              metrics=_good_metrics(), universe=UNIVERSE)
    assert out is not None
    assert out["source"] == "deepseek-llm"
    assert len(out["claims"]) == 1
    assert out["claims"][0]["confidence"] == 0.65
    assert out["_meta"]["input_tokens"] == 1234
    assert out["_meta"]["output_tokens"] == 567


def test_llm_interpret_returns_none_on_http_error(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-test")
    def _raise(*a, **kw):
        raise TimeoutError("simulated timeout")
    with patch.object(L.urllib.request, "urlopen", side_effect=_raise):
        out = L.llm_interpret(asof="2026-01-01", signals=_good_signals(),
                              metrics=_good_metrics())
    assert out is None
