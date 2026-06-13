"""Devil's Advocate agent — the discipline layer.

Reads the FULL analysis (fundamental + valuation + risk + tactical + catalysts +
hedge) and aggressively looks for reasons the recommendation is WRONG.

Per the investment framework, Ch.6 (投資心理學), the biggest enemy of an investor is their
own bias. This agent exists specifically to be that adversary. It will reject
recommendations that look superficially correct but have a fatal flaw — exactly
the kind of "永久性資本損失" that the 3 super-rules ("Don't die") try to prevent.

Outputs:
  - overall_verdict: pass / pass_with_concerns / veto
  - findings: list of specific issues with category, severity, evidence
  - counter_thesis: the strongest bear/contrarian case for the position
"""
from __future__ import annotations

from typing import Optional

from src.agents.base import fmt_optional, fmt_pct
from src.llm.client import chat_json
from src.models.schemas import (
    AnalysisResult,
    DevilAdvocateReview,
    DevilFinding,
)


SYSTEM_PROMPT = """You are a senior risk officer at a top-tier hedge fund. Your job is to be the adversary
of the recommendation in front of you.

The other agents have built a case. Your job: TRY TO BREAK IT.

You are anchored in our consolidated investment framework:
- Don't die (capital preservation, margin of safety, risk management, diversification)
- Let time compound (long horizon, liquidity premium)
- Know your limits (behavioral finance, circle of competence, valuation discipline, contrarian on evidence)

Specifically scan for these failure modes:
  1. confirmation_bias — does the thesis cherry-pick? what data points are absent?
  2. anchoring — is the analysis anchored on cost basis, 52w high, prior price?
  3. overconfidence — is the valuation confidence band too narrow given the visibility?
  4. narrative_fallacy — is the story too neat? real businesses are messier
  5. circle_of_competence — is the business actually understandable, or are we pretending?
  6. moat_erosion — what's the evidence the moat is INTACT (not just historical)?
  7. macro_blind_spot — what macro/regime shift could invalidate this thesis?
  8. behavioral_fomo — are we buying because it's been working, or because of fundamentals?
  9. data_quality — thin financials, conflicting numbers, missing peer data?
  10. valuation_optimism — what if growth is half what's modeled?
  11. risk_underestimation — what tail risk is being ignored?
  12. technical_fundamental_contradiction — does the price-action picture (structure,
       volume, relative strength) DISAGREE with the fundamental call?
       Examples that MUST be flagged:
         • Fundamental says BUY_NOW but technical composite is bearish/strong_bearish,
           structure is in a confirmed downtrend, and volume shows distribution.
         • Fundamental quality is high and MoS is positive, but RS shows the name is
           a weak_laggard vs both sector and index for 90d + 365d.
         • Tactical says trim/exit but technical composite is bullish with accumulation
           — the system is selling INTO buying pressure.
       The Technical Division was deliberately BLINDED from fundamentals so its signal
       is an independent check. Disagreements are signal, not noise.

For EACH finding:
  - category: from the list above
  - severity:
       info     = noteworthy but not action-changing
       concern  = should weight against the recommendation; user should be aware
       veto     = fatal flaw; the recommendation must NOT be acted on
  - finding: 1 sentence stating the problem
  - evidence: 1-2 sentences with the specific data point or omission
  - recommendation: what should be done about it (e.g., wait for catalyst, demand more data, reduce size, abandon)

Overall verdict logic:
  - veto: ANY finding with severity=veto, OR three or more concern-level findings on independent failure modes
  - pass_with_concerns: 1-2 concern-level findings
  - pass: only info-level findings, or none

counter_thesis: 2-3 sentences making the STRONGEST bear case for this position.
Imagine you're a portfolio manager arguing AGAINST the analyst who proposed this trade.

Be honest and specific. Generic objections ("market could fall") are not findings.
Cite specific numbers, ratios, or omissions. If you can't find real issues, return overall_verdict=pass
with an empty findings list and an honest counter_thesis. Do not invent flaws to look thorough.
"""


def _format_user(ar: AnalysisResult, prior_audit_summary: Optional[str] = None) -> str:
    f = ar.fundamental
    v = ar.valuation
    r = ar.risk
    pos = ar.position
    held_action = ar.if_held.tactical.action
    held_label = ar.if_held.tactical.label or "NO_ACTION"

    pos_block = (
        f"  Shares: {pos.shares}\n"
        f"  Cost basis/share: {pos.cost_basis_per_share} {pos.currency}\n"
        f"  Days held: {(ar.timestamp_utc.date() - pos.purchase_date).days if pos.purchase_date else 'unknown'}"
        if pos
        else "  (not currently held)"
    )

    cats_block = "  (none)"
    if ar.forward_catalysts and ar.forward_catalysts.key_catalysts:
        cats_block = "\n".join(
            f"  - [{c.direction}/{c.confidence}] {c.event} ({c.expected_date or 'no date'})"
            for c in ar.forward_catalysts.key_catalysts[:6]
        )

    hedge_block = "  (no hedge plan)"
    if ar.hedge_plan and ar.hedge_plan.candidates:
        rec = ar.hedge_plan.candidates[ar.hedge_plan.recommended_index]
        hedge_block = f"  Recommended: short {rec.instrument} (corr {rec.correlation_90d})"

    prior = f"\nPRIOR-RUN CONTEXT:\n{prior_audit_summary}" if prior_audit_summary else ""

    # Technical Division (Phase 6) — blinded independent signal
    tech_block = "  (Technical Division did not run)"
    if ar.technical is not None:
        t = ar.technical
        tech_block = (
            f"  Composite: {t.composite_signal.upper()}  ({t.composite_rationale})\n"
            f"  Structure: trend={t.structure.trend} stage={t.structure.stage} "
            f"(confidence {t.structure.confidence:.2f})\n"
            f"  Volume: {t.volume.institutional_flow} / OBV {t.volume.obv_trend} "
            f"(20vs50d {t.volume.volume_expansion_pct:+.0f}%, UDV/DDV {t.volume.up_down_volume_ratio})\n"
            f"  Cost-basis: trapped supply {t.cost_basis.trapped_supply_pct:.0f}%, "
            f"accumulation {t.cost_basis.accumulation_pct:.0f}%, {len(t.cost_basis.hvn_levels)} HVN level(s)\n"
            f"  Relative strength: {t.relative_strength.signal} "
            f"(vs sector 90d/365d = {t.relative_strength.vs_sector_etf_90d}/"
            f"{t.relative_strength.vs_sector_etf_365d}; "
            f"vs index 90d/365d = {t.relative_strength.vs_index_90d}/"
            f"{t.relative_strength.vs_index_365d})\n"
            f"  Price-map key_support={t.price_map.key_support}, "
            f"key_resistance={t.price_map.key_resistance}"
        )

    return f"""SYMBOL: {ar.symbol}  ({ar.market})
CURRENT PRICE: {ar.current_price:.2f} {ar.currency}
TIMESTAMP: {ar.timestamp_utc.isoformat()}

POSITION:
{pos_block}

FUNDAMENTAL ASSESSMENT (from Fundamental Analyst):
  Quality score: {f.quality_score:.1f}/10
  Moat strength: {f.moat_strength}  ({f.moat_assessment})
  Balance sheet: {f.balance_sheet_health}
  Growth outlook: {f.growth_outlook}
  Capital allocation: {f.capital_allocation}
  Red flags: {'; '.join(f.red_flags) or '(none reported)'}
  ROIC: {fmt_pct(f.roic_pct)}  Op Margin: {fmt_pct(f.operating_margin_pct)}  D/E: {fmt_optional(f.debt_to_equity)}
  Thesis: {f.thesis_one_liner}

VALUATION:
  Intrinsic range: {v.intrinsic_low:.2f} – {v.intrinsic_base:.2f} – {v.intrinsic_high:.2f} {v.currency}
  Margin of safety vs current: {v.margin_of_safety_pct:+.1f}%
  Confidence: {v.confidence}
  Methodology: {v.methodology_notes}
  DCF value: {fmt_optional(v.dcf_value)} | Multiples value: {fmt_optional(v.multiples_value)}

RISK ASSESSMENT (over {r.horizon_days}d horizon):
  Realized vol (annualized): {r.realized_vol_annualized_pct:.1f}%
  P(drawdown ≥ 10/15/20/25%): {r.drawdown_probabilities}
  Key macro signals: {'; '.join(r.key_macro_signals) or '(none)'}
  Scenarios:
{chr(10).join(f'    - {s.name} ({s.probability*100:.0f}%): ret {s.expected_return_pct:+.1f}%, dd {s.expected_drawdown_pct:+.1f}% — {s.rationale}' for s in r.scenarios)}

FORWARD CATALYSTS (next 30d):
{cats_block}
  Sentiment score: {ar.forward_catalysts.sentiment_score if ar.forward_catalysts else 'n/a'}

TACTICAL RECOMMENDATION:
  Level: {held_label}
  Action: {held_action}
  Rationale: {ar.if_held.tactical.rationale}

HEDGE PLAN:
{hedge_block}

IF NOT HELD RECOMMENDATION: {ar.if_not_held.recommendation}
  Rationale: {ar.if_not_held.rationale}

TECHNICAL DIVISION (INDEPENDENT — these agents were blinded from the fundamental side):
{tech_block}{prior}

Apply the failure-mode checklist from the system prompt. Be specific and adversarial.
Use real numbers and named omissions. If the recommendation is genuinely sound, say so —
do not manufacture concerns.
"""


def review(
    ar: AnalysisResult,
    prior_audit_summary: Optional[str] = None,
    use_reasoner: bool = True,
) -> DevilAdvocateReview:
    """Run Devil's Advocate review on a completed AnalysisResult."""
    user_msg = _format_user(ar, prior_audit_summary)
    result = chat_json(
        system=SYSTEM_PROMPT,
        user=user_msg,
        schema=DevilAdvocateReview,
        reasoner=use_reasoner,
        temperature=0.3,
    )

    # Enforce the verdict logic deterministically (LLM may stray)
    veto_findings = [f for f in result.findings if f.severity == "veto"]
    concern_findings = [f for f in result.findings if f.severity == "concern"]
    if veto_findings:
        result.overall_verdict = "veto"
        if not result.veto_reason:
            result.veto_reason = "; ".join(f.finding for f in veto_findings[:3])
    elif len(concern_findings) >= 3:
        result.overall_verdict = "veto"
        if not result.veto_reason:
            result.veto_reason = f"{len(concern_findings)} independent concerns: " + "; ".join(
                f.category for f in concern_findings[:5]
            )
    elif concern_findings:
        result.overall_verdict = "pass_with_concerns"
    else:
        result.overall_verdict = "pass"

    return result


def apply_veto(ar: AnalysisResult, dr: DevilAdvocateReview) -> bool:
    """Hard-enforce a DA veto on a completed AnalysisResult.

    The architecture diagram claims DA "can VETO"; this function makes
    that claim load-bearing. When `dr.overall_verdict == "veto"`:
      - if_held.immediate_orders and rebuy_orders are cleared
      - if_not_held.recommendation is forced to PASS and entry_orders cleared
      - rationales are prepended with a [DEVIL'S ADVOCATE VETO] marker and the reason

    Idempotent. Returns True if a veto was applied, False otherwise.
    The tactical action label is left intact (so the dashboard still shows
    "would have been ORANGE_TRIM"); only the actionable orders are stripped.
    """
    if dr.overall_verdict != "veto":
        return False

    reason = dr.veto_reason or dr.summary or "Devil's Advocate veto."
    marker = "[DEVIL'S ADVOCATE VETO]"
    short = reason if len(reason) <= 300 else reason[:297] + "…"

    # Held path: clear actionable orders, keep tactical-label visible
    ar.if_held.immediate_orders = []
    ar.if_held.rebuy_orders = []
    if marker not in ar.if_held.tactical.rationale:
        ar.if_held.tactical.rationale = (
            f"{marker} {short} Orders cleared. Original: {ar.if_held.tactical.rationale}"
        )

    # Not-held path: force PASS, clear entry orders
    original_rec = ar.if_not_held.recommendation
    ar.if_not_held.recommendation = "PASS"
    ar.if_not_held.entry_orders = []
    if marker not in ar.if_not_held.rationale:
        ar.if_not_held.rationale = (
            f"{marker} {short} Recommendation forced to PASS "
            f"(was {original_rec}). Original: {ar.if_not_held.rationale}"
        )

    return True
