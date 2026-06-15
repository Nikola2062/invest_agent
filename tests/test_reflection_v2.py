"""Tests for Phase 4 — multi-horizon outcomes store + statistical aggregator.

These cover the failure modes I worried about in the original review:

  1. Outcomes store must persist multi-horizon alpha correctly and
     refuse duplicates (so re-runs don't double-count trades).
  2. Aggregator must NOT emit a lesson when n < min_trades — that's the
     whole point: silence is the right output for thin data.
  3. Aggregator must NOT emit a lesson when |t-stat| < threshold — even
     with 1000 trades, if the mean alpha is indistinguishable from zero,
     the PM should hear nothing.
  4. When a real signal is present (consistent positive alpha on a
     bucket), the aggregator surfaces it with the right direction marker
     and quoted statistics.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tradingagents.portfolio.outcomes_store import (
    DEFAULT_HORIZONS,
    OutcomeRow,
    OutcomesStore,
)
from tradingagents.portfolio.reflection_v2 import (
    BucketStat,
    DEFAULT_MIN_TRADES,
    DEFAULT_T_THRESHOLD,
    aggregate_by_rating_sector,
    render_calibrated_lessons,
)

pytestmark = pytest.mark.unit


# --- outcomes store --------------------------------------------------------


def _row(ticker, date, rating, sector="Tech", alpha_5d=None, alpha_21d=None, alpha_63d=None):
    return OutcomeRow(
        ticker=ticker, trade_date=date, rating=rating, sector=sector,
        horizons={
            5: {"raw": None, "alpha": alpha_5d},
            21: {"raw": None, "alpha": alpha_21d},
            63: {"raw": None, "alpha": alpha_63d},
        },
    )


def test_store_writes_and_reads_back(tmp_path):
    store = OutcomesStore(tmp_path / "out.csv")
    store.append(_row("AAPL", "2024-01-15", "Buy", alpha_5d=0.02, alpha_21d=0.05))
    rows = store.load()
    assert len(rows) == 1
    assert rows[0]["ticker"] == "AAPL"
    assert float(rows[0]["alpha_5d"]) == pytest.approx(0.02)
    assert float(rows[0]["alpha_21d"]) == pytest.approx(0.05)
    assert rows[0]["alpha_63d"] in ("", None)


def test_store_rejects_duplicates(tmp_path):
    store = OutcomesStore(tmp_path / "out.csv")
    assert store.append(_row("AAPL", "2024-01-15", "Buy", alpha_5d=0.02)) is True
    assert store.append(_row("AAPL", "2024-01-15", "Buy", alpha_5d=0.04)) is False
    rows = store.load()
    assert len(rows) == 1


def test_store_update_overwrites_existing(tmp_path):
    """Later runs that resolve longer horizons must overwrite, not duplicate."""
    store = OutcomesStore(tmp_path / "out.csv")
    store.append(_row("AAPL", "2024-01-15", "Buy", alpha_5d=0.02))
    store.update(_row("AAPL", "2024-01-15", "Buy", alpha_5d=0.02, alpha_21d=0.06, alpha_63d=0.12))
    rows = store.load()
    assert len(rows) == 1
    assert float(rows[0]["alpha_21d"]) == pytest.approx(0.06)
    assert float(rows[0]["alpha_63d"]) == pytest.approx(0.12)


def test_store_default_horizons():
    s = OutcomesStore("/tmp/_unused.csv")
    assert s.horizons == DEFAULT_HORIZONS


# --- aggregation -----------------------------------------------------------


def test_aggregator_skips_hold_ratings():
    rows = [
        {"rating": "Hold", "sector": "Tech", "alpha_21d": 0.01},
        {"rating": "Buy", "sector": "Tech", "alpha_21d": 0.02},
    ]
    stats = aggregate_by_rating_sector(rows, horizon_days=21)
    assert all(s.rating != "Hold" for s in stats)


def test_aggregator_skips_unresolved_horizons():
    rows = [
        {"rating": "Buy", "sector": "Tech", "alpha_21d": ""},
        {"rating": "Buy", "sector": "Tech", "alpha_21d": None},
        {"rating": "Buy", "sector": "Tech", "alpha_21d": 0.03},
    ]
    stats = aggregate_by_rating_sector(rows, horizon_days=21)
    assert len(stats) == 1
    assert stats[0].n == 1  # only one resolved row


def test_aggregator_computes_hit_rate_correctly_for_long_ratings():
    rows = [
        {"rating": "Buy", "sector": "Tech", "alpha_21d": 0.05},
        {"rating": "Buy", "sector": "Tech", "alpha_21d": 0.02},
        {"rating": "Buy", "sector": "Tech", "alpha_21d": -0.01},
        {"rating": "Buy", "sector": "Tech", "alpha_21d": -0.03},
    ]
    stats = aggregate_by_rating_sector(rows, horizon_days=21)
    assert stats[0].hit_rate == pytest.approx(0.5)  # 2 of 4 positive


def test_aggregator_computes_hit_rate_correctly_for_short_ratings():
    rows = [
        {"rating": "Sell", "sector": "Energy", "alpha_21d": -0.05},
        {"rating": "Sell", "sector": "Energy", "alpha_21d": -0.02},
        {"rating": "Sell", "sector": "Energy", "alpha_21d": 0.01},
    ]
    stats = aggregate_by_rating_sector(rows, horizon_days=21)
    assert stats[0].hit_rate == pytest.approx(2 / 3)  # 2 of 3 went down


def test_aggregator_sorts_by_descending_abs_t_stat():
    rows = (
        # High t-stat bucket: tight positive alpha, n=25
        [{"rating": "Buy", "sector": "Tech", "alpha_21d": 0.05} for _ in range(25)]
        # Low t-stat bucket: noisy alpha around zero, n=25
        + [{"rating": "Buy", "sector": "Energy", "alpha_21d": 0.01 - 0.02 * (i % 2)}
           for i in range(25)]
    )
    stats = aggregate_by_rating_sector(rows, horizon_days=21)
    assert stats[0].sector == "Tech"
    assert abs(stats[0].t_stat) > abs(stats[1].t_stat)


# --- significance gating — the critical Phase 4 contract ------------------


def test_no_lesson_below_min_trades(tmp_path):
    """A bucket with 5 trades must not emit a calibrated lesson."""
    store = OutcomesStore(tmp_path / "out.csv")
    for i in range(5):
        store.append(_row(f"T{i}", f"2024-0{i + 1}-15", "Buy", alpha_21d=0.10))

    block = render_calibrated_lessons(
        store, horizon_days=21, min_trades=20, t_threshold=1.5,
    )
    # Falls through to the "showing current leaders for context" preview,
    # which explicitly states no significance was reached.
    assert "no bucket has reached significance" in block


def test_no_lesson_when_alpha_centred_on_zero(tmp_path):
    """50 trades of pure noise → no significant lesson."""
    store = OutcomesStore(tmp_path / "out.csv")
    # Symmetric alpha distribution: mean is 0, t-stat is 0.
    for i in range(50):
        sign = 1 if i % 2 == 0 else -1
        store.append(_row(
            f"T{i}", f"2024-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
            "Buy", alpha_21d=sign * 0.02,
        ))
    block = render_calibrated_lessons(
        store, horizon_days=21, min_trades=20, t_threshold=1.5,
    )
    assert "no bucket has reached significance" in block


def test_lesson_emitted_with_real_signal(tmp_path):
    """30 trades of consistently positive alpha → significant lesson surfaces."""
    store = OutcomesStore(tmp_path / "out.csv")
    # Mean +2%, std ~1% → t = 0.02 / (0.01/sqrt(30)) ≈ 11. Well above gate.
    for i in range(30):
        alpha = 0.02 + 0.01 * ((i % 3) - 1)  # values: 0.01, 0.02, 0.03
        store.append(_row(
            f"T{i}", f"2024-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
            "Buy", alpha_21d=alpha,
        ))
    block = render_calibrated_lessons(
        store, horizon_days=21, min_trades=20, t_threshold=1.5,
    )
    assert "Calibrated lessons" in block
    assert "Buy (Tech)" in block
    assert "n=30" in block
    # Directional ✓ marker (Buy + positive alpha).
    assert "✓" in block


def test_lesson_flags_wrong_direction(tmp_path):
    """Consistent NEGATIVE alpha on Buy ratings → ✗ direction marker."""
    store = OutcomesStore(tmp_path / "out.csv")
    for i in range(30):
        store.append(_row(
            f"T{i}", f"2024-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
            "Buy", alpha_21d=-0.03,
        ))
    block = render_calibrated_lessons(
        store, horizon_days=21, min_trades=20, t_threshold=1.5,
    )
    assert "✗" in block
    assert "Buy (Tech)" in block


def test_lesson_block_empty_when_store_missing(tmp_path):
    """No store → empty block, not a crash."""
    store = OutcomesStore(tmp_path / "out.csv")  # never written
    assert render_calibrated_lessons(store) == ""


def test_max_lines_caps_output(tmp_path):
    """The lessons block should not blow up the prompt with 50+ buckets."""
    store = OutcomesStore(tmp_path / "out.csv")
    # 12 buckets, each with 30 trades of strong signal.
    sectors = ["Tech", "Health", "Energy", "Finance", "Consumer", "Industrial",
               "Utilities", "Materials", "Comm", "RealEstate", "Staples", "Other"]
    for sec in sectors:
        for i in range(30):
            store.append(_row(
                f"{sec}_{i}", f"2024-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
                "Buy", sector=sec, alpha_21d=0.04,
            ))
    block = render_calibrated_lessons(
        store, horizon_days=21, min_trades=20, t_threshold=1.5, max_lines=5,
    )
    # Header + up to max_lines bullet lines.
    bullet_lines = [ln for ln in block.splitlines() if ln.startswith("- ")]
    assert len(bullet_lines) <= 5


# --- BucketStat formatting ------------------------------------------------


def test_bucket_stat_lesson_line_format():
    stat = BucketStat(
        rating="Buy", sector="Tech", horizon_days=21,
        n=30, hit_rate=0.7, mean_alpha=0.025, std_alpha=0.04, t_stat=3.42,
    )
    line = stat.to_lesson_line()
    assert "Buy (Tech)" in line
    assert "n=30" in line
    assert "70%" in line
    assert "t=+3.42" in line


def test_bucket_stat_significance_gates():
    sig = BucketStat("Buy", "Tech", 21, n=25, hit_rate=0.6, mean_alpha=0.02, std_alpha=0.03, t_stat=2.0)
    assert sig.is_significant() is True

    too_few = BucketStat("Buy", "Tech", 21, n=10, hit_rate=0.7, mean_alpha=0.02, std_alpha=0.03, t_stat=2.0)
    assert too_few.is_significant() is False

    too_noisy = BucketStat("Buy", "Tech", 21, n=50, hit_rate=0.5, mean_alpha=0.001, std_alpha=0.05, t_stat=0.5)
    assert too_noisy.is_significant() is False


# --- integration: memory log injection ------------------------------------


def test_memory_log_injects_calibrated_block_when_enabled(tmp_path):
    """get_past_context appends the calibrated block when config opts in."""
    from tradingagents.agents.utils.memory import TradingMemoryLog

    outcomes_path = tmp_path / "outcomes.csv"
    store = OutcomesStore(outcomes_path)
    for i in range(30):
        store.append(_row(
            f"T{i}", f"2024-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
            "Buy", alpha_21d=0.03,
        ))

    config = {
        "memory_log_path": str(tmp_path / "memory.md"),
        "calibrated_reflection_enabled": True,
        "outcomes_store_path": str(outcomes_path),
        "data_cache_dir": str(tmp_path),
    }
    log = TradingMemoryLog(config)
    context = log.get_past_context("AAPL")
    assert "Calibrated lessons" in context


def test_memory_log_skips_calibrated_block_when_disabled(tmp_path):
    """When the flag is off, the calibrated block doesn't appear even if data exists."""
    from tradingagents.agents.utils.memory import TradingMemoryLog

    outcomes_path = tmp_path / "outcomes.csv"
    store = OutcomesStore(outcomes_path)
    for i in range(30):
        store.append(_row(
            f"T{i}", f"2024-{(i % 12) + 1:02d}-{(i % 27) + 1:02d}",
            "Buy", alpha_21d=0.03,
        ))

    config = {
        "memory_log_path": str(tmp_path / "memory.md"),
        "calibrated_reflection_enabled": False,
        "outcomes_store_path": str(outcomes_path),
        "data_cache_dir": str(tmp_path),
    }
    log = TradingMemoryLog(config)
    assert "Calibrated lessons" not in log.get_past_context("AAPL")
