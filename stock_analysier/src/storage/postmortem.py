"""Post-mortem reporting off the audit trail.

For each historical recommendation, fetch the current price and compute the
outcome. Aggregates by recommendation type, by symbol, and overall.

Pure deterministic. No LLM. Output is a PostMortemReport dataclass.
"""
from __future__ import annotations

import logging
import sqlite3
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional

import yfinance as yf

from src.storage.audit import DEFAULT_DB_PATH

log = logging.getLogger(__name__)

Outcome = Literal["correct", "wrong", "neutral", "pending"]


@dataclass
class RecommendationOutcome:
    symbol: str
    timestamp: datetime
    recommendation_type: str          # e.g. "BUY_NOW", "PASS", "YELLOW_WATCH", "devil_veto_BUY_NOW"
    price_at_recommendation: float
    current_price: float
    days_elapsed: int
    pct_change: float
    outcome: Outcome
    explanation: str


@dataclass
class CategoryStats:
    correct: int = 0
    wrong: int = 0
    neutral: int = 0
    pending: int = 0

    @property
    def total(self) -> int:
        return self.correct + self.wrong + self.neutral + self.pending

    @property
    def hit_rate(self) -> float:
        decided = self.correct + self.wrong
        return self.correct / decided if decided else 0.0


@dataclass
class PostMortemReport:
    period_start: datetime
    period_end: datetime
    total_recommendations: int
    by_type: dict[str, CategoryStats] = field(default_factory=dict)
    by_symbol: dict[str, CategoryStats] = field(default_factory=dict)
    outcomes: list[RecommendationOutcome] = field(default_factory=list)
    calibration_overall: float = 0.0  # decided correct / decided total
    notable_winners: list[RecommendationOutcome] = field(default_factory=list)
    notable_losers: list[RecommendationOutcome] = field(default_factory=list)
    summary: str = ""


# Outcome thresholds (tunable)
_MIN_DAYS_TO_JUDGE_BUY = 7        # BUY_NOW needs at least 7 days
_MIN_DAYS_TO_JUDGE_PASS = 14      # PASS needs at least 2 weeks of "stock didn't run"
_MIN_DAYS_TO_JUDGE_EXIT = 7
_BUY_CORRECT_PCT = 2.0            # +2% or more = correct
_BUY_WRONG_PCT = -5.0             # -5% or worse = wrong
_PASS_CORRECT_PCT = 0.0           # flat or down = correct
_PASS_WRONG_PCT = 10.0            # +10% or more after PASS = wrong (we missed it)
_EXIT_CORRECT_PCT = -3.0          # stock fell after exit = correct
_EXIT_WRONG_PCT = 5.0             # stock kept rising after exit = wrong


def _judge(rec_type: str, pct_change: float, days_elapsed: int) -> tuple[Outcome, str]:
    """Decide whether a historical recommendation looks right in hindsight."""
    if rec_type in ("BUY_NOW",):
        if days_elapsed < _MIN_DAYS_TO_JUDGE_BUY:
            return "pending", f"Only {days_elapsed}d elapsed — need ≥ {_MIN_DAYS_TO_JUDGE_BUY}d to judge."
        if pct_change >= _BUY_CORRECT_PCT:
            return "correct", f"Stock +{pct_change:.1f}% since BUY_NOW recommendation."
        if pct_change <= _BUY_WRONG_PCT:
            return "wrong", f"Stock {pct_change:.1f}% since BUY_NOW — entry timing was poor."
        return "neutral", f"Stock {pct_change:+.1f}% — within noise."

    if rec_type in ("WAIT_FOR_PRICE",):
        if days_elapsed < _MIN_DAYS_TO_JUDGE_BUY:
            return "pending", f"Only {days_elapsed}d elapsed."
        # WAIT_FOR_PRICE: "right" means stock either came down (allowing entry) OR was overvalued and is now lower
        if pct_change <= 0:
            return "correct", f"Stock {pct_change:.1f}% — wait paid off (better entry available)."
        if pct_change >= _PASS_WRONG_PCT:
            return "wrong", f"Stock +{pct_change:.1f}% — we left the trade on the table."
        return "neutral", f"Stock {pct_change:+.1f}%."

    if rec_type == "PASS":
        if days_elapsed < _MIN_DAYS_TO_JUDGE_PASS:
            return "pending", f"Only {days_elapsed}d elapsed."
        if pct_change <= _PASS_CORRECT_PCT:
            return "correct", f"Stock {pct_change:+.1f}% after PASS — avoided exposure."
        if pct_change >= _PASS_WRONG_PCT:
            return "wrong", f"Stock +{pct_change:.1f}% after PASS — missed a real opportunity."
        return "neutral", f"Stock {pct_change:+.1f}%."

    # Tactical exits / trims (held positions)
    if rec_type in ("ORANGE_TRIM", "RED_DEFENSIVE", "BLACK_EXIT"):
        if days_elapsed < _MIN_DAYS_TO_JUDGE_EXIT:
            return "pending", f"Only {days_elapsed}d elapsed."
        if pct_change <= _EXIT_CORRECT_PCT:
            return "correct", f"Stock {pct_change:.1f}% after defensive action — drawdown captured."
        if pct_change >= _EXIT_WRONG_PCT:
            return "wrong", f"Stock +{pct_change:.1f}% after exit — premature defensive sell."
        return "neutral", f"Stock {pct_change:+.1f}%."

    if rec_type == "YELLOW_WATCH":
        # Monitor-only: never "wrong" since no action, just informational
        return "neutral", "YELLOW_WATCH is monitor-only — no action taken."

    if rec_type == "no_action":
        if days_elapsed < _MIN_DAYS_TO_JUDGE_PASS:
            return "pending", f"Only {days_elapsed}d elapsed."
        if pct_change <= -15:
            return "wrong", f"Held through {pct_change:.1f}% drawdown — risk policy may have missed a signal."
        return "correct", f"Held through {pct_change:+.1f}% — no defensive action was warranted."

    # Phase 6: Technical Division composite calls
    if rec_type in ("tech_strong_bullish", "tech_bullish"):
        if days_elapsed < _MIN_DAYS_TO_JUDGE_BUY:
            return "pending", f"Only {days_elapsed}d elapsed — need ≥ {_MIN_DAYS_TO_JUDGE_BUY}d."
        if pct_change >= _BUY_CORRECT_PCT:
            return "correct", f"Composite was bullish → stock {pct_change:+.1f}% — confirmed."
        if pct_change <= _BUY_WRONG_PCT:
            return "wrong", f"Composite was bullish → stock {pct_change:.1f}% — disconfirmed."
        return "neutral", f"Composite bullish → {pct_change:+.1f}% (within noise)."

    if rec_type in ("tech_strong_bearish", "tech_bearish"):
        if days_elapsed < _MIN_DAYS_TO_JUDGE_EXIT:
            return "pending", f"Only {days_elapsed}d elapsed."
        if pct_change <= _EXIT_CORRECT_PCT:
            return "correct", f"Composite was bearish → stock {pct_change:.1f}% — confirmed."
        if pct_change >= _EXIT_WRONG_PCT:
            return "wrong", f"Composite was bearish → stock +{pct_change:.1f}% — disconfirmed."
        return "neutral", f"Composite bearish → {pct_change:+.1f}% (within noise)."

    if rec_type == "tech_neutral":
        return "neutral", "Composite was neutral — no directional call to grade."

    return "neutral", "Recommendation type does not have a backtest rule."


def _current_price(symbol: str) -> Optional[float]:
    try:
        df = yf.Ticker(symbol).history(period="5d")["Close"].dropna()
        if df.empty:
            return None
        return float(df.iloc[-1])
    except Exception as e:
        log.warning("yf price fetch failed for %s: %s", symbol, e)
        return None


def _classify_row(row: sqlite3.Row, current_price: float, now_utc: datetime) -> RecommendationOutcome:
    """Pick the single most-informative recommendation type for this audit row."""
    ts = datetime.fromisoformat(row["timestamp_utc"])
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    days = max(0, (now_utc - ts).days)
    price_then = float(row["current_price"])
    pct = (current_price / price_then - 1) * 100 if price_then else 0.0

    # Priority: tactical_label > if_not_held_recommendation > if_held_action
    if row["tactical_label"]:
        rec_type = row["tactical_label"]
    elif row["if_not_held_recommendation"]:
        rec_type = row["if_not_held_recommendation"]
    else:
        rec_type = row["if_held_action"] or "no_action"

    outcome, explanation = _judge(rec_type, pct, days)
    return RecommendationOutcome(
        symbol=row["symbol"],
        timestamp=ts,
        recommendation_type=rec_type,
        price_at_recommendation=price_then,
        current_price=current_price,
        days_elapsed=days,
        pct_change=pct,
        outcome=outcome,
        explanation=explanation,
    )


def _classify_technical(row: sqlite3.Row, current_price: float, now_utc: datetime) -> Optional[RecommendationOutcome]:
    """Phase 6: produce a separate outcome entry for the technical composite signal,
    so the post-mortem can grade how well the Technical Division called direction
    INDEPENDENTLY of the fundamental side.
    """
    try:
        signal = row["composite_tech_signal"]
    except (IndexError, KeyError):
        return None
    if not signal:
        return None
    ts = datetime.fromisoformat(row["timestamp_utc"])
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    days = max(0, (now_utc - ts).days)
    price_then = float(row["current_price"])
    pct = (current_price / price_then - 1) * 100 if price_then else 0.0
    rec_type = f"tech_{signal}"
    outcome, explanation = _judge(rec_type, pct, days)
    return RecommendationOutcome(
        symbol=row["symbol"],
        timestamp=ts,
        recommendation_type=rec_type,
        price_at_recommendation=price_then,
        current_price=current_price,
        days_elapsed=days,
        pct_change=pct,
        outcome=outcome,
        explanation=explanation,
    )


_NARRATOR_SYSTEM = """You are an objective calibration reviewer.

You receive a deterministic post-mortem report: counts of correct/wrong/pending recommendations
broken down by type (BUY_NOW, PASS, tactical exits, etc.) plus notable winners and losers.

Your job: write 1-2 paragraphs (max ~150 words) that answer:
  - Is the system biased? In which direction (too bullish? too defensive? too quick to PASS?)
  - Where is it most reliable? Where most unreliable?
  - Are there patterns in the misses (one sector? one ticker? one rec-type?)
  - What should the user TUNE? (e.g., raise MoS threshold for BUY_NOW; lower drawdown_jump threshold)

Be specific. Cite hit rates. Don't generic-coach. If the sample size is too small for conclusions,
say so explicitly — small-sample bias is real.
"""


def narrate(report: "PostMortemReport") -> str:
    """Optional LLM-written qualitative calibration summary."""
    from src.llm.client import chat_text  # lazy import to avoid LLM dep at module load
    if not report.outcomes:
        return "No outcomes to narrate yet."
    decided = sum(s.correct + s.wrong for s in report.by_type.values())
    if decided < 3:
        return f"Sample too small ({decided} decided outcomes). Build more history before drawing conclusions."

    type_lines = "\n".join(
        f"  - {t}: total={s.total} correct={s.correct} wrong={s.wrong} pending={s.pending} "
        f"hit_rate={s.hit_rate*100:.0f}%"
        for t, s in report.by_type.items()
    )
    winners = "\n".join(
        f"  - {o.symbol} {o.recommendation_type} {o.timestamp.date()}: {o.pct_change:+.1f}%"
        for o in report.notable_winners
    ) or "  (none)"
    losers = "\n".join(
        f"  - {o.symbol} {o.recommendation_type} {o.timestamp.date()}: {o.pct_change:+.1f}%"
        for o in report.notable_losers
    ) or "  (none)"

    user_msg = f"""POST-MORTEM SUMMARY ({report.period_start.date()} → {report.period_end.date()}):
Total recommendations: {report.total_recommendations}
Decided: {decided}  Calibration: {report.calibration_overall*100:.0f}%

By recommendation type:
{type_lines}

Notable winners:
{winners}

Notable losers:
{losers}

Write a 1-2 paragraph calibration assessment per the system prompt.
"""
    try:
        return chat_text(system=_NARRATOR_SYSTEM, user=user_msg, temperature=0.3)
    except Exception as e:
        log.warning("Narrator failed: %s", e)
        return f"(Narrator unavailable: {e})"


def generate_report(
    since_days: int = 90,
    db_path: Path = DEFAULT_DB_PATH,
    with_narrator: bool = False,
) -> PostMortemReport:
    """Build a PostMortemReport across all audit rows in the trailing N days."""
    now = datetime.now(timezone.utc)

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(
            "SELECT * FROM analysis_runs WHERE timestamp_utc >= datetime('now', ?) ORDER BY timestamp_utc",
            (f"-{since_days} days",),
        )
        rows = list(cur.fetchall())

    if not rows:
        return PostMortemReport(
            period_start=now,
            period_end=now,
            total_recommendations=0,
            summary="No audit history in the requested window.",
        )

    # Cache current prices per symbol so we don't refetch
    symbols = sorted({r["symbol"] for r in rows})
    price_cache: dict[str, Optional[float]] = {s: _current_price(s) for s in symbols}

    outcomes: list[RecommendationOutcome] = []
    for row in rows:
        cp = price_cache.get(row["symbol"])
        if cp is None:
            continue
        outcomes.append(_classify_row(row, cp, now))
        tech_outcome = _classify_technical(row, cp, now)
        if tech_outcome is not None:
            outcomes.append(tech_outcome)

    by_type: dict[str, CategoryStats] = defaultdict(CategoryStats)
    by_symbol: dict[str, CategoryStats] = defaultdict(CategoryStats)
    for o in outcomes:
        setattr(by_type[o.recommendation_type], o.outcome, getattr(by_type[o.recommendation_type], o.outcome) + 1)
        setattr(by_symbol[o.symbol], o.outcome, getattr(by_symbol[o.symbol], o.outcome) + 1)

    decided_correct = sum(s.correct for s in by_type.values())
    decided_total = sum(s.correct + s.wrong for s in by_type.values())
    calibration = decided_correct / decided_total if decided_total else 0.0

    # Notable: top winners / losers by absolute pct_change among DECIDED outcomes
    decided = [o for o in outcomes if o.outcome in ("correct", "wrong")]
    decided_sorted = sorted(decided, key=lambda o: o.pct_change, reverse=True)
    winners = [o for o in decided_sorted if o.outcome == "correct"][:5]
    losers = [o for o in decided_sorted if o.outcome == "wrong"][-5:]

    first_ts = datetime.fromisoformat(rows[0]["timestamp_utc"])
    if first_ts.tzinfo is None:
        first_ts = first_ts.replace(tzinfo=timezone.utc)

    summary = (
        f"{len(outcomes)} recommendations in last {since_days}d. "
        f"Decided: {decided_total} ({decided_correct} correct, "
        f"{decided_total - decided_correct} wrong) — calibration {calibration*100:.0f}%."
    )

    report = PostMortemReport(
        period_start=first_ts,
        period_end=now,
        total_recommendations=len(outcomes),
        by_type=dict(by_type),
        by_symbol=dict(by_symbol),
        outcomes=outcomes,
        calibration_overall=calibration,
        notable_winners=winners,
        notable_losers=losers,
        summary=summary,
    )
    if with_narrator:
        report.summary = f"{summary}\n\n--- Narrator ---\n{narrate(report)}"
    return report


def print_report(since_days: int = 90) -> None:
    """CLI entry — pretty-prints the report."""
    r = generate_report(since_days=since_days)
    print(f"\n=== Post-Mortem Report ({r.period_start.date()} → {r.period_end.date()}) ===")
    print(r.summary)
    if not r.outcomes:
        return
    print(f"\nBy recommendation type:")
    for rec_type, stats in sorted(r.by_type.items(), key=lambda x: -x[1].total):
        decided = stats.correct + stats.wrong
        rate = f"{stats.hit_rate*100:.0f}%" if decided else "—"
        print(f"  {rec_type:20s}  total={stats.total:3d}  correct={stats.correct:3d}  wrong={stats.wrong:3d}  pending={stats.pending:3d}  hit-rate={rate}")

    print(f"\nNotable winners:")
    for o in r.notable_winners:
        print(f"  ✅ {o.symbol} {o.recommendation_type} {o.timestamp.date()} → {o.pct_change:+.1f}% in {o.days_elapsed}d")
    print(f"\nNotable losers:")
    for o in r.notable_losers:
        print(f"  ❌ {o.symbol} {o.recommendation_type} {o.timestamp.date()} → {o.pct_change:+.1f}% in {o.days_elapsed}d")


if __name__ == "__main__":
    import sys
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 90
    print_report(since_days=days)
