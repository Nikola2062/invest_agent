"""Tests for the Competence Gate — deterministic keyword/sector matching."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.competence_gate import _keyword_matches, assess


def test_keyword_word_boundaries():
    # "AI" should NOT match "retailer" (substring)
    assert _keyword_matches("AI", "Walmart is a discount retailer") is False
    # "AI" SHOULD match when it's a standalone word
    assert _keyword_matches("AI", "NVIDIA builds AI accelerators") is True
    assert _keyword_matches("AI", "AI is the future.") is True
    # Multi-word
    assert _keyword_matches("artificial intelligence", "We use artificial intelligence here.") is True
    # Case-insensitive
    assert _keyword_matches("software", "SOFTWARE company") is True
    print("✓ Word-boundary keyword matching is correct")


def test_in_circle_via_sector():
    v = assess("MSFT", "Technology", "Microsoft sells cloud software and productivity tools.")
    assert v.verdict == "in_circle", v
    assert any("Technology" in c for c in v.matched_categories)
    print("✓ Sector match → in_circle")


def test_in_circle_via_keyword():
    v = assess("FIG", "Software—Application", "Figma is a design platform for collaborative product teams.")
    # Sector 'Software—Application' may or may not match; relying on keywords
    assert v.verdict == "in_circle", v
    assert any("kw:in" in c for c in v.matched_categories)
    print("✓ Keyword match → in_circle")


def test_out_of_circle_sector():
    v = assess("XOM", "Energy", "ExxonMobil is an integrated oil and gas company.")
    assert v.verdict == "out_of_circle"
    print("✓ Out-of-circle sector → out_of_circle")


def test_out_of_circle_keyword():
    v = assess(
        "BIOX", "Technology",
        "BioX runs clinical trials for novel pharmaceutical drug pipelines.",
    )
    assert v.verdict == "out_of_circle"
    assert any("kw:out" in c for c in v.matched_categories)
    print("✓ Out-of-circle keyword overrides Technology sector")


def test_borderline_when_no_match():
    v = assess("WMT", "Consumer Defensive", "Walmart is a discount retailer operating supercenters.")
    assert v.verdict == "borderline"
    print("✓ No matches → borderline")


def test_override_in_circle():
    # BRK.B is in always_in_circle even though sector is Financial Services (not on in-circle list)
    v = assess("BRK.B", "Financial Services", "Berkshire Hathaway holds diversified businesses.")
    assert v.verdict == "in_circle"
    assert any("override" in c for c in v.matched_categories)
    print("✓ always_in_circle override wins")


def test_insufficient_data_when_both_empty():
    """Both sector AND description empty → distinct 'insufficient_data' verdict.

    Without this branch, a yfinance fetch failure silently produces a
    'borderline — no keywords matched' result that looks identical to a real
    outside-circle call. Surface the data gap explicitly.
    """
    v = assess("FOO", None, None)
    assert v.verdict == "borderline"
    assert "insufficient_data" in v.matched_categories
    assert "fetch" in v.reasoning.lower() or "data" in v.reasoning.lower()
    print("✓ Both empty → borderline with insufficient_data marker")


def test_insufficient_data_empty_strings():
    """Empty strings should be treated the same as None."""
    v = assess("FOO", "", "   ")
    assert v.verdict == "borderline"
    assert "insufficient_data" in v.matched_categories


def test_override_wins_even_when_data_missing():
    """always_in_circle override fires before the insufficient_data check —
    so a known-good ticker classifies correctly even on a fetch failure."""
    v = assess("BRK.B", None, None)
    assert v.verdict == "in_circle"
    assert any("override" in c for c in v.matched_categories)


def test_sector_only_does_not_trigger_insufficient_data():
    """A non-empty sector alone is enough data for the keyword/sector logic."""
    v = assess("FOO", "Technology", None)
    assert "insufficient_data" not in v.matched_categories


if __name__ == "__main__":
    test_keyword_word_boundaries()
    test_in_circle_via_sector()
    test_in_circle_via_keyword()
    test_out_of_circle_sector()
    test_out_of_circle_keyword()
    test_borderline_when_no_match()
    test_override_in_circle()
    test_insufficient_data_when_both_empty()
    test_insufficient_data_empty_strings()
    test_override_wins_even_when_data_missing()
    test_sector_only_does_not_trigger_insufficient_data()
    print("\nAll competence gate tests passed ✅")
