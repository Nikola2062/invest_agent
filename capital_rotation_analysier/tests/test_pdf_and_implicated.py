"""Tests for the PDF renderer and the Section-12 implicated-products feature."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from rotation.interpret import NARRATIVE_BASKETS, interpret
from rotation.report import _check_observed_direction, section_explanations


# ---------- implicated products ----------

def test_narrative_baskets_cover_all_narratives():
    """Every narrative builder should have a basket defined (no silent gaps)."""
    # All narratives that emit a Claim must appear in NARRATIVE_BASKETS.
    expected = {
        "risk_on_rotation", "risk_off_rotation", "growth_acceleration",
        "inflation_rising", "inflation_easing", "recession_concern",
        "liquidity_tightening", "unusual_volume",
    }
    missing = expected - set(NARRATIVE_BASKETS.keys())
    assert not missing, f"narratives without basket: {missing}"


def test_interpret_attaches_implicated_products():
    """When risk-off fires, the claim must carry implicated_strengthening
    (defensives) and implicated_weakening (risk assets)."""
    signals = {
        "risk_on_off": {"score": -50, "confidence": 0.65},
        "recession":   {"score": 40, "confidence": 0.85},
        "capital_rotation": {"score": -30, "confidence": 0.70},
    }
    out = interpret(signals)
    risk_off = next((c for c in out["claims"] if c["narrative_id"] == "risk_off_rotation"), None)
    assert risk_off is not None
    assert "TLT" in risk_off["implicated_strengthening"]
    assert "GLD" in risk_off["implicated_strengthening"]
    assert "SPY" in risk_off["implicated_weakening"]
    assert "BTC-USD" in risk_off["implicated_weakening"]


def test_check_observed_direction_marks_consistent():
    """A ticker the narrative says should strengthen, with positive r_w, marks ✓."""
    metrics = pd.DataFrame([
        {"symbol": "TLT", "r_w": 0.02, "r_d": 0.005},
        {"symbol": "SPY", "r_w": -0.03, "r_d": -0.01},
        {"symbol": "FLAT", "r_w": 0.001, "r_d": 0.0},
    ])
    mark_tlt, _ = _check_observed_direction("TLT", expected_strengthens=True, metrics=metrics)
    assert mark_tlt == "✓"

    mark_spy, _ = _check_observed_direction("SPY", expected_strengthens=False, metrics=metrics)
    assert mark_spy == "✓", "SPY weakened and narrative predicted weakening — should be ✓"


def test_check_observed_direction_marks_contradicting():
    metrics = pd.DataFrame([
        {"symbol": "IWM", "r_w": 0.015, "r_d": 0.01},
    ])
    # Narrative predicts IWM weakens; it actually strengthened
    mark, _ = _check_observed_direction("IWM", expected_strengthens=False, metrics=metrics)
    assert mark == "✗"


def test_check_observed_direction_marks_flat_as_mixed():
    metrics = pd.DataFrame([
        {"symbol": "UUP", "r_w": 0.001, "r_d": 0.0},
    ])
    mark, _ = _check_observed_direction("UUP", expected_strengthens=True, metrics=metrics)
    assert mark == "—"


def test_section_explanations_renders_implicated_products():
    """When a narrative fires and metrics are passed, the rendered section must
    include 'Products implicated' with per-ticker marks."""
    signals = {
        "risk_on_off": {"score": -50, "confidence": 0.65},
        "recession":   {"score": 40, "confidence": 0.85},
    }
    out = interpret(signals)
    metrics = pd.DataFrame([
        {"symbol": "TLT", "r_w": 0.02, "r_d": 0.005},
        {"symbol": "SPY", "r_w": -0.03, "r_d": -0.01},
    ])
    rendered = section_explanations(out, metrics)
    assert "Products implicated" in rendered
    assert "TLT ✓" in rendered
    assert "SPY ✓" in rendered


# ---------- PDF renderer ----------

def test_pdf_renderer_writes_a_valid_pdf(tmp_path):
    from rotation.pdf_report import markdown_to_pdf
    md = (
        "# Test report\n\n"
        "Some text.\n\n"
        "| col | val |\n|---|---:|\n| a | 1.23 |\n| b | 4.56 |\n"
    )
    out = tmp_path / "test.pdf"
    result = markdown_to_pdf(md, out)
    assert result.exists()
    assert result.stat().st_size > 1000, "PDF should be > 1KB even for trivial content"
    # PDFs begin with %PDF-
    head = result.read_bytes()[:5]
    assert head == b"%PDF-", f"expected PDF signature, got {head!r}"
