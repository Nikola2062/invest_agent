"""Statistical aggregator over the outcomes store.

The original ``Reflector.reflect_on_final_decision`` writes prose for
every single trade — and given a random 5-day outcome plus a thesis,
an LLM will reliably confabulate a plausible "lesson". Inject those into
the next prompt and the PM is learning from noise.

This aggregator does the opposite: it computes statistics over the
*pool* of past trades, then emits a calibrated lesson only when the
bucket's t-statistic clears a threshold. The rest of the time it stays
silent — silence is the right output when there's no signal.

Default thresholds are deliberately liberal (n≥20, |t|≥1.5). These
aren't publication-grade significance levels; the goal is to nudge the
PM's prompt with directional patterns when they exist, not to gate
trading decisions through them. Tighten the thresholds if you want the
PM to see only high-confidence patterns; loosen them for more lessons
at the cost of more false positives.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Iterable, Optional

from .outcomes_store import OutcomesStore

logger = logging.getLogger(__name__)


# Sensible defaults for "should this bucket emit a lesson?". n_min keeps
# us from teaching the PM patterns from 3 trades; t_min is a soft
# significance gate. They're tunable via render_calibrated_lessons().
DEFAULT_MIN_TRADES = 20
DEFAULT_T_THRESHOLD = 1.5
# The horizon used for the headline statistic. 21d is the standard
# equity-research lookahead and is far enough out that hit rates aren't
# pure noise — 5d hit rates of a random strategy hover at 50% by chance.
DEFAULT_HEADLINE_HORIZON = 21


@dataclass
class BucketStat:
    """Aggregated statistics for one (rating, sector) bucket at one horizon."""

    rating: str
    sector: str
    horizon_days: int
    n: int
    hit_rate: float          # fraction of trades with alpha in the predicted direction
    mean_alpha: float        # mean alpha at this horizon
    std_alpha: float         # std of alpha at this horizon
    t_stat: float            # mean / (std / sqrt(n))

    def is_significant(
        self,
        min_trades: int = DEFAULT_MIN_TRADES,
        t_threshold: float = DEFAULT_T_THRESHOLD,
    ) -> bool:
        if self.n < min_trades:
            return False
        return abs(self.t_stat) >= t_threshold

    def to_lesson_line(self) -> str:
        """One-line markdown lesson for prompt injection."""
        direction = "✓" if self._is_directionally_correct() else "✗"
        return (
            f"- **{direction} {self.rating} ({self.sector})** "
            f"over {self.horizon_days}d: "
            f"n={self.n}, hit={self.hit_rate * 100:.0f}%, "
            f"mean α={self.mean_alpha * 100:+.2f}%, t={self.t_stat:+.2f}"
        )

    def _is_directionally_correct(self) -> bool:
        """True when the mean alpha matches the rating's direction.

        Buy/Overweight → expect positive alpha.
        Sell/Underweight → expect negative alpha.
        Hold → directional correctness is undefined; treat as ✓ (no claim).
        """
        if self.rating in ("Buy", "Overweight"):
            return self.mean_alpha > 0
        if self.rating in ("Sell", "Underweight"):
            return self.mean_alpha < 0
        return True


# --- core stats helpers --------------------------------------------------


def _mean_std(values: list[float]) -> tuple[float, float]:
    n = len(values)
    if n == 0:
        return 0.0, 0.0
    m = sum(values) / n
    if n < 2:
        return m, 0.0
    var = sum((v - m) ** 2 for v in values) / (n - 1)  # sample variance
    return m, math.sqrt(var)


def _t_stat(values: list[float]) -> float:
    """One-sample t against zero.

    Returns 0 when undefined (too few values, or all zero). When the
    sample is degenerate but consistent — std=0, mean≠0 — the t-stat
    is mathematically infinite, and we return ``copysign(inf, mean)``
    so the significance gate correctly classifies it as significant
    rather than silently rounding to zero.
    """
    n = len(values)
    if n < 2:
        return 0.0
    m, sd = _mean_std(values)
    if sd == 0:
        if m == 0:
            return 0.0
        return math.copysign(math.inf, m)
    return m / (sd / math.sqrt(n))


def _hit_for_rating(rating: str, alpha: float) -> bool:
    """Trade hits when alpha matches the rating's directional bet.

    Hold is neither bullish nor bearish; we don't count it as a hit/miss.
    """
    if rating in ("Buy", "Overweight"):
        return alpha > 0
    if rating in ("Sell", "Underweight"):
        return alpha < 0
    return False


# --- aggregation ---------------------------------------------------------


def aggregate_by_rating_sector(
    rows: Iterable[dict],
    horizon_days: int = DEFAULT_HEADLINE_HORIZON,
) -> list[BucketStat]:
    """Group rows by (rating, sector) and compute stats at one horizon.

    Excludes Hold ratings (no direction to score) and rows where the
    given horizon is unresolved (alpha is None / empty). Sorted by
    descending |t-stat| so the most-significant bucket appears first.
    """
    col = f"alpha_{horizon_days}d"
    buckets: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        rating = r.get("rating", "")
        if rating in ("", "Hold"):
            continue
        sector = r.get("sector") or "Unknown"
        val = r.get(col)
        if val in (None, "", "None"):
            continue
        try:
            alpha = float(val)
        except (TypeError, ValueError):
            continue
        buckets.setdefault((rating, sector), []).append(alpha)

    out: list[BucketStat] = []
    for (rating, sector), values in buckets.items():
        if not values:
            continue
        mean, std = _mean_std(values)
        n_hits = sum(1 for v in values if _hit_for_rating(rating, v))
        hit_rate = n_hits / len(values)
        out.append(BucketStat(
            rating=rating, sector=sector, horizon_days=horizon_days,
            n=len(values), hit_rate=hit_rate,
            mean_alpha=mean, std_alpha=std,
            t_stat=_t_stat(values),
        ))
    out.sort(key=lambda s: abs(s.t_stat), reverse=True)
    return out


def render_calibrated_lessons(
    store: OutcomesStore,
    *,
    horizon_days: int = DEFAULT_HEADLINE_HORIZON,
    min_trades: int = DEFAULT_MIN_TRADES,
    t_threshold: float = DEFAULT_T_THRESHOLD,
    max_lines: int = 8,
) -> str:
    """Read the store, compute stats, and render a prompt-ready lessons block.

    Returns the empty string when no bucket clears the significance gate
    — silence is the correct behaviour when there's no calibrated signal.
    The caller is responsible for deciding *whether* to inject this into
    the PM prompt (gate it on config).
    """
    rows = store.load()
    if not rows:
        return ""

    stats = aggregate_by_rating_sector(rows, horizon_days=horizon_days)
    significant = [
        s for s in stats if s.is_significant(min_trades, t_threshold)
    ][:max_lines]

    if not significant:
        # Surface the next-most-significant buckets at zero confidence so the
        # PM at least knows what the *data* looks like, even when nothing
        # passes the gate. Limit to 3 so we don't dilute the prompt.
        if not stats:
            return ""
        preview = stats[:3]
        lines = [
            "**Calibrated lessons:** _no bucket has reached significance "
            f"(need n≥{min_trades}, |t|≥{t_threshold}); showing current "
            "leaders for context:_",
        ]
        lines.extend(s.to_lesson_line() for s in preview)
        return "\n".join(lines)

    header = (
        f"**Calibrated lessons** (horizon {horizon_days}d, "
        f"n≥{min_trades}, |t|≥{t_threshold}; sorted by |t|):"
    )
    lines = [header] + [s.to_lesson_line() for s in significant]
    return "\n".join(lines)
