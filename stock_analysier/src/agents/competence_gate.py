"""Competence Gate agent — pure deterministic.

Pre-filters tickers against the user's declared circle of competence
(config/competence.yaml). Per the investment framework, Ch.12: knowing what you DON'T
know is more valuable than knowing more.

Verdict is purely sector + keyword + symbol-override match. No LLM. Cheap enough
to run on every ticker.
"""
from __future__ import annotations

import re
from typing import Optional

from src.config.loader import load_competence
from src.models.schemas import CompetenceVerdict


def _keyword_matches(keyword: str, text: str) -> bool:
    """Word-boundary keyword match (case-insensitive).

    Avoids the "AI" → "retAIler" substring trap. Multi-word keywords (e.g.
    "artificial intelligence") match as a phrase.
    """
    if not keyword or not text:
        return False
    pattern = r"\b" + re.escape(keyword.strip()) + r"\b"
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def assess(
    symbol: str,
    sector: Optional[str],
    business_description: Optional[str],
) -> CompetenceVerdict:
    cfg = load_competence()

    # 1. Symbol-level overrides take precedence (work even when data fetch failed)
    if symbol in cfg.always_in_circle:
        return CompetenceVerdict(
            symbol=symbol,
            verdict="in_circle",
            reasoning="Symbol explicitly whitelisted in always_in_circle.",
            matched_categories=["override:always_in_circle"],
        )
    if symbol in cfg.always_out_of_circle:
        return CompetenceVerdict(
            symbol=symbol,
            verdict="out_of_circle",
            reasoning="Symbol explicitly blacklisted in always_out_of_circle.",
            matched_categories=["override:always_out_of_circle"],
        )

    matched: list[str] = []
    desc = business_description or ""
    sec = (sector or "").strip()

    # 1b. Insufficient data → distinct borderline verdict. Without this, a yfinance
    # hiccup that returns an empty sector + description would silently classify as
    # "no in-circle match found" — indistinguishable from a real "outside circle"
    # call. Surfacing the data gap lets the dashboard show a fetch warning.
    if not sec and not desc.strip():
        return CompetenceVerdict(
            symbol=symbol,
            verdict="borderline",
            reasoning=(
                "Insufficient fundamentals data — sector AND business description "
                "are both empty (likely upstream fetch failure). Cannot classify "
                "against the configured circle of competence. Verify manually."
            ),
            matched_categories=["insufficient_data"],
        )

    # 2. Out-of-circle wins over in-circle on conflict
    if sec and sec in cfg.out_of_circle_sectors:
        return CompetenceVerdict(
            symbol=symbol,
            verdict="out_of_circle",
            reasoning=f"Sector '{sec}' is in out_of_circle_sectors.",
            matched_categories=[f"sector:out:{sec}"],
        )

    out_kw_hits = [kw for kw in cfg.out_of_circle_keywords if _keyword_matches(kw, desc)]
    if out_kw_hits:
        return CompetenceVerdict(
            symbol=symbol,
            verdict="out_of_circle",
            reasoning=f"Business description contains out-of-circle keyword(s): {', '.join(out_kw_hits)}.",
            matched_categories=[f"kw:out:{k}" for k in out_kw_hits],
        )

    # 3. In-circle matching
    if sec and sec in cfg.in_circle_sectors:
        matched.append(f"sector:in:{sec}")
    in_kw_hits = [kw for kw in cfg.in_circle_keywords if _keyword_matches(kw, desc)]
    matched.extend(f"kw:in:{k}" for k in in_kw_hits)

    if matched:
        verdict = "in_circle"
        reasoning = f"Matches: {', '.join(matched[:5])}"
    else:
        verdict = "borderline"
        reasoning = (
            f"Sector '{sec or 'unknown'}' is not on the in-circle list, "
            f"and no in-circle keywords matched the business description. "
            f"Proceed with caution — analysis runs, but treat the result as low-confidence."
        )

    return CompetenceVerdict(
        symbol=symbol,
        verdict=verdict,
        reasoning=reasoning,
        matched_categories=matched,
    )


def skip_pipeline_for(verdict: CompetenceVerdict) -> bool:
    """Return True if the on_out_of_circle policy says to skip the full pipeline."""
    if verdict.verdict != "out_of_circle":
        return False
    cfg = load_competence()
    return cfg.on_out_of_circle.get("policy") == "skip"
