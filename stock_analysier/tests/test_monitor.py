"""Tests for the Monitor agent — pure deterministic trigger logic.

Uses a tempfile SQLite DB so the real audit.sqlite isn't touched.
"""
from __future__ import annotations

import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from datetime import datetime, timedelta, timezone

from src.agents.monitor import _detect_for_symbol, _detect_rebuy_stale


def _row(d: dict) -> sqlite3.Row:
    """Build a sqlite3.Row from a dict for testing."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    cols = ", ".join(d.keys())
    placeholders = ", ".join("?" * len(d))
    conn.execute(f"CREATE TABLE r ({cols})")
    conn.execute(f"INSERT INTO r VALUES ({placeholders})", list(d.values()))
    return conn.execute("SELECT * FROM r").fetchone()


_BASELINE = {
    "symbol": "FIG", "timestamp_utc": "2026-05-01T12:00:00",
    "current_price": 30.0,
    "quality_score": 8.0, "margin_of_safety_pct": 25.0,
    "p_dd_15": 0.20, "p_dd_20": 0.10,
    "tactical_label": None, "if_held_action": "no_action",
    "if_not_held_recommendation": "WAIT_FOR_PRICE",
    "sentiment_score": 0.30, "realized_vol_pct": 50.0,
    "devil_verdict": "pass",
}


def _override(**kwargs):
    d = dict(_BASELINE)
    d.update(kwargs)
    return _row(d)


def test_no_break_on_steady_state():
    prev = _row(_BASELINE)
    curr = _row({**_BASELINE, "timestamp_utc": "2026-05-02T12:00:00"})
    breaks = _detect_for_symbol(curr, prev)
    assert breaks == [], f"expected no breaks, got {[b.category for b in breaks]}"
    print("✓ Steady state → no breaks")


def test_quality_drop():
    prev = _row(_BASELINE)
    curr = _override(quality_score=5.5)  # 8.0 → 5.5 = -2.5
    breaks = _detect_for_symbol(curr, prev)
    cats = [b.category for b in breaks]
    assert "quality_drop" in cats, cats
    print("✓ Quality drop ≥ 2pts → quality_drop alert")


def test_mos_flip():
    prev = _row(_BASELINE)
    curr = _override(margin_of_safety_pct=-10.0)  # +25 → -10
    breaks = _detect_for_symbol(curr, prev)
    assert any(b.category == "mos_flip" and b.severity == "critical" for b in breaks)
    print("✓ MoS flip → critical mos_flip alert")


def test_drawdown_jump():
    prev = _row(_BASELINE)
    curr = _override(p_dd_20=0.35)  # +25pp
    breaks = _detect_for_symbol(curr, prev)
    assert any(b.category == "drawdown_jump" for b in breaks)
    print("✓ P(dd≥20%) jump ≥ 20pp → drawdown_jump alert")


def test_tactical_escalation_to_black():
    prev = _row(_BASELINE)  # tactical_label = None
    curr = _override(tactical_label="BLACK_EXIT")  # 4 steps up
    breaks = _detect_for_symbol(curr, prev)
    assert any(b.category == "tactical_escalation" and b.severity == "critical" for b in breaks), [
        (b.category, b.severity) for b in breaks
    ]
    print("✓ Escalation to BLACK_EXIT → critical tactical_escalation alert")


def test_devil_veto_flip():
    prev = _row({**_BASELINE, "devil_verdict": "pass"})
    curr = _override(devil_verdict="veto")
    breaks = _detect_for_symbol(curr, prev)
    assert any(b.category == "devil_veto" and b.severity == "critical" for b in breaks)
    print("✓ Devil's Advocate verdict flip to veto → critical devil_veto alert")


def test_sentiment_drop():
    prev = _row(_BASELINE)  # sentiment 0.30
    curr = _override(sentiment_score=-0.30)  # -0.60 delta
    breaks = _detect_for_symbol(curr, prev)
    assert any(b.category == "sentiment_drop" for b in breaks)
    print("✓ Sentiment drop ≥ 0.5 → sentiment_drop alert")


def test_vol_spike():
    prev = _row(_BASELINE)  # vol 50%
    curr = _override(realized_vol_pct=80.0)  # +60% relative
    breaks = _detect_for_symbol(curr, prev)
    assert any(b.category == "vol_spike" for b in breaks)
    print("✓ Realized vol spike +50% relative → vol_spike alert")


def test_multiple_breaks_stacked():
    prev = _row(_BASELINE)
    curr = _override(
        quality_score=5.0,         # quality_drop
        margin_of_safety_pct=-10,   # mos_flip
        p_dd_20=0.40,               # drawdown_jump
        tactical_label="BLACK_EXIT",  # tactical_escalation
        devil_verdict="veto",       # devil_veto
    )
    breaks = _detect_for_symbol(curr, prev)
    cats = {b.category for b in breaks}
    assert cats >= {"quality_drop", "mos_flip", "drawdown_jump", "tactical_escalation", "devil_veto"}, cats
    print(f"✓ Multiple simultaneous breaks → {len(breaks)} alerts of types {sorted(cats)}")


def test_technical_signal_flip_bullish_to_bearish():
    prev = _override(composite_tech_signal="bullish")
    curr = _override(composite_tech_signal="bearish")
    breaks = _detect_for_symbol(curr, prev)
    cats = [b.category for b in breaks]
    assert "technical_signal_flip" in cats, cats
    print("✓ Composite flip bullish→bearish → technical_signal_flip alert")


def test_technical_signal_flip_no_alert_on_drift():
    prev = _override(composite_tech_signal="neutral")
    curr = _override(composite_tech_signal="bearish")
    breaks = _detect_for_symbol(curr, prev)
    cats = [b.category for b in breaks]
    assert "technical_signal_flip" not in cats, "should not fire on neutral→bearish drift"
    print("✓ Neutral → bearish does NOT trigger flip alert (no polarity reversal)")


def _trim_row(days_ago: int, rebuy_high: float, label: str = "ORANGE_TRIM", current: float = 60.0) -> sqlite3.Row:
    """Synthetic audit row representing a past trim/exit event."""
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    body = json.dumps({"if_held": {"tactical": {"rebuy_band_high": rebuy_high, "label": label}}})
    return _row({**_BASELINE, "timestamp_utc": ts, "tactical_label": label,
                 "current_price": current, "full_result_json": body})


def _no_action_row(days_ago: int, current: float) -> sqlite3.Row:
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return _row({**_BASELINE, "timestamp_utc": ts, "tactical_label": None,
                 "current_price": current, "full_result_json": json.dumps({"if_held": {"tactical": {}}})})


def test_rebuy_stale_fires_when_band_unreached():
    """Trim 45 days ago, rebuy band ≤ $54, current price $60 — should fire."""
    rows = [
        _no_action_row(days_ago=0, current=60.0),    # most recent run (current price 60)
        _trim_row(days_ago=45, rebuy_high=54.0),     # old trim, band 54
    ]
    br = _detect_rebuy_stale("FIG", rows, staleness_days=30)
    assert br is not None, "expected rebuy_stale alert"
    assert br.category == "rebuy_stale"
    assert br.evidence["elapsed_days"] >= 30
    print("✓ Stale rebuy (45d, price above band) → rebuy_stale alert")


def test_rebuy_stale_silent_when_within_band():
    """Trim 45 days ago, but current price $52 sits inside band ≤ $54 — no alert."""
    rows = [
        _no_action_row(days_ago=0, current=52.0),
        _trim_row(days_ago=45, rebuy_high=54.0),
    ]
    br = _detect_rebuy_stale("FIG", rows, staleness_days=30)
    assert br is None, f"expected no alert (price within band), got {br}"
    print("✓ Price within rebuy band → no rebuy_stale alert")


def test_rebuy_stale_silent_before_threshold():
    """Trim only 10 days ago — below 30-day threshold, no alert yet."""
    rows = [
        _no_action_row(days_ago=0, current=60.0),
        _trim_row(days_ago=10, rebuy_high=54.0),
    ]
    br = _detect_rebuy_stale("FIG", rows, staleness_days=30)
    assert br is None, f"expected no alert (under threshold), got {br}"
    print("✓ Recent trim (10d) → no rebuy_stale alert yet")


if __name__ == "__main__":
    test_no_break_on_steady_state()
    test_quality_drop()
    test_mos_flip()
    test_drawdown_jump()
    test_tactical_escalation_to_black()
    test_devil_veto_flip()
    test_sentiment_drop()
    test_vol_spike()
    test_multiple_breaks_stacked()
    test_rebuy_stale_fires_when_band_unreached()
    test_rebuy_stale_silent_when_within_band()
    test_rebuy_stale_silent_before_threshold()
    print("\nAll monitor tests passed ✅")
