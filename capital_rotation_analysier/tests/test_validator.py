"""Tests for ingest/validator.py — must catch the OHLC sanity violations
that distinguish a legitimate large move from a split/dividend not yet applied."""
from __future__ import annotations

from datetime import date

from rotation.ingest.validator import is_stale, validate_row


def _good_row(**kw):
    base = {
        "symbol": "SPY", "asset_class": "equity_us", "ts": date(2025, 6, 1),
        "open": 100.0, "high": 102.0, "low": 99.0, "close": 101.0,
        "adj_close": 101.0, "volume": 1_000_000.0, "source": "test",
    }
    base.update(kw); return base


def test_validate_row_accepts_good():
    ok, reason = validate_row(_good_row(), outlier_intraday_pct=20.0)
    assert ok, reason
    assert reason == ""


def test_validate_row_rejects_negative_price():
    ok, reason = validate_row(_good_row(close=-1.0), outlier_intraday_pct=20.0)
    assert not ok
    assert "negative" in reason


def test_validate_row_rejects_missing_field():
    ok, reason = validate_row(_good_row(low=None), outlier_intraday_pct=20.0)
    assert not ok
    assert "missing" in reason


def test_validate_row_rejects_high_below_close():
    # High must be >= max(open, close, low)
    ok, reason = validate_row(_good_row(high=99.0, close=101.0), outlier_intraday_pct=20.0)
    assert not ok


def test_validate_row_rejects_low_above_open():
    ok, reason = validate_row(_good_row(low=110.0, high=120.0, close=115.0, open=105.0), outlier_intraday_pct=20.0)
    assert not ok


def test_validate_row_rejects_outlier_intraday_range():
    # 50% intraday range > 20% threshold -> quarantined
    ok, reason = validate_row(_good_row(low=50.0, high=110.0, close=100.0, open=100.0),
                              outlier_intraday_pct=20.0)
    assert not ok
    assert "intraday_range" in reason


def test_validate_row_allows_outlier_at_higher_threshold():
    # Same row, but threshold relaxed for crypto/commodities
    ok, _ = validate_row(_good_row(low=50.0, high=110.0, close=100.0, open=100.0),
                         outlier_intraday_pct=80.0)
    assert ok


def test_is_stale_boundary():
    asof = date(2025, 6, 10)
    # 4 calendar days ago is the threshold (per config default)
    assert not is_stale(date(2025, 6, 6), asof, threshold_days=4)  # exactly 4 days
    assert is_stale(date(2025, 6, 5), asof, threshold_days=4)       # 5 days, stale
