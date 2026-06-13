"""Tests for the independent Hong Kong report (report_hk.py)."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from rotation.config import (
    Config, IngestConfig, RetryPolicy, StorageConfig, Symbol,
)
from rotation.report_hk import (
    build_hk_daily_report, compute_hk_metrics, load_hk_panels,
)
from rotation.store import connect, upsert_bars


def _cfg(tmp_path) -> Config:
    return Config(
        storage=StorageConfig(duckdb_path=tmp_path / "rot.duckdb"),
        ingest=IngestConfig(
            primary_source="yfinance",
            retry=RetryPolicy(max_attempts=1, backoff_seconds=(1,)),
            stale_threshold_days=4,
            outlier_intraday_pct=20.0,
            outlier_intraday_pct_by_class={},
        ),
        universe=(
            Symbol("0700.HK", "equity_hk"),
            Symbol("2800.HK", "equity_hk"),
            Symbol("0939.HK", "equity_hk"),
        ),
    )


def _seed_hk_bars(cfg: Config, asof: date, n_days: int = 90) -> None:
    """Weekday bars: 0700.HK trends up, 2800.HK trends down, 0939.HK flat."""
    rows = []
    d = asof
    i = 0
    while i < n_days:
        if d.weekday() < 5:
            for sym, base, drift in (("0700.HK", 400.0, +0.004),
                                     ("2800.HK", 25.0, -0.004),
                                     ("0939.HK", 9.0, 0.0)):
                px = base * float(np.exp(drift * (n_days - i)))
                rows.append({
                    "symbol": sym, "asset_class": "equity_hk", "ts": d,
                    "open": px, "high": px * 1.01, "low": px * 0.99,
                    "close": px, "adj_close": px, "volume": 1e6,
                    "source": "test", "ingested_at": datetime(2026, 1, 1),
                    "stale": False,
                })
            i += 1
        d -= timedelta(days=1)
    with connect(cfg.storage.duckdb_path) as con:
        upsert_bars(con, rows)


def test_compute_hk_metrics_shape_and_ranks(tmp_path):
    cfg = _cfg(tmp_path)
    asof = date(2026, 6, 9)
    _seed_hk_bars(cfg, asof)
    with connect(cfg.storage.duckdb_path) as con:
        close, volume = load_hk_panels(con, asof)
    m = compute_hk_metrics(close, volume)

    assert set(m["symbol"]) == {"0700.HK", "2800.HK", "0939.HK"}
    for col in ("r_d", "r_w", "r_m", "rs_rank", "rs_change_1",
                "rs_change_5", "rs_accel_5"):
        assert col in m.columns
    by = m.set_index("symbol")
    # Uptrending name must out-rank downtrending one; flat sits between.
    assert by.loc["0700.HK", "rs_rank"] > by.loc["0939.HK", "rs_rank"] \
           > by.loc["2800.HK", "rs_rank"]
    assert by.loc["0700.HK", "r_w"] > 0 > by.loc["2800.HK", "r_w"]


def test_build_hk_daily_report_sections(tmp_path):
    cfg = _cfg(tmp_path)
    asof = date(2026, 6, 9)
    _seed_hk_bars(cfg, asof)
    body = build_hk_daily_report(cfg, asof)

    assert body.startswith("# Hong Kong Rotation Report")
    # Same 27-section skeleton as the US report (+ appendix).
    for i, title in [
        (1, "Overview"), (2, "Investment Committee View"), (3, "Market Regime"),
        (4, "Capital Flow Dashboard"), (5, "Flow Map"),
        (6, "Leadership Rotation Tracker"), (7, "Rotation Strength"),
        (8, "Capital Rotation — Pair Breakdown"), (9, "Historical Analogues"),
        (10, "Regime Transition Probabilities"), (11, "Where Money Likely Goes Next"),
        (12, "Probabilistic Market Forecast"), (13, "Sector Forecast"),
        (14, "Forecast Scorecard"),
        (15, "Top Strengthening Assets"), (16, "Top Weakening Assets"),
        (17, "Sector / Bucket Breadth"), (18, "Volume Anomalies"),
        (19, "ETF Flow Analysis"), (20, "Detected Themes"),
        (21, "Signal Attribution"),
        (22, "What Changed Since Yesterday"), (23, "What Changed Since Last Week"),
        (24, "What Changed Since Last Month"), (25, "Potential Explanations"),
        (26, "Confidence Assessment"), (27, "Glossary"),
    ]:
        assert f"## Section {i} — {title}" in body, f"missing section {i}: {title}"
    assert "## Appendix — Per-Ticker Detail" in body
    assert "<NA>" not in body and "nan%" not in body


def test_build_hk_report_empty_store(tmp_path):
    cfg = _cfg(tmp_path)
    with connect(cfg.storage.duckdb_path):
        pass  # create schema only
    body = build_hk_daily_report(cfg, date(2026, 6, 9))
    assert "No HK bars" in body  # graceful, no crash


def test_hk_etf_flows_renders_when_rows_present(tmp_path):
    """D5: when 2800.HK / 2828.HK rows exist in etf_flows, §17 shows the table
    instead of the legacy 'not snapshotted yet' placeholder."""
    cfg = _cfg(tmp_path)
    asof = date(2026, 6, 9)
    _seed_hk_bars(cfg, asof, n_days=90)

    # Seed a couple of etf_flows rows the way flows_adapter would.
    import duckdb
    from rotation.ingest.flows_adapter import upsert_flows
    with connect(cfg.storage.duckdb_path) as con:
        upsert_flows(con, [
            {"symbol": "2800.HK", "ts": asof, "shares_outstanding": 3_155_990_016.0,
             "aum_usd": 78_584_150_194.0, "net_flow_usd": None,
             "source": "yfinance", "proxy_method": "shares_delta", "confidence": 0.6},
            {"symbol": "2828.HK", "ts": asof, "shares_outstanding": 341_563_475.0,
             "aum_usd": 29_333_470_320.0, "net_flow_usd": None,
             "source": "yfinance", "proxy_method": "shares_delta", "confidence": 0.6},
        ])

    body = build_hk_daily_report(cfg, asof)
    # Section 19 renders the actual table, not the legacy placeholder.
    assert "## Section 19 — ETF Flow Analysis" in body
    assert "2800.HK" in body and "2828.HK" in body
    assert "Flow history accumulated:" in body
    assert "not snapshotted yet" not in body


def test_hk_etf_flows_no_rows_shows_placeholder(tmp_path):
    """When etf_flows has no HK rows yet, §17 shows the post-D5 placeholder
    referencing the auto-populate path, not the legacy 'covers US only' wording."""
    cfg = _cfg(tmp_path)
    asof = date(2026, 6, 9)
    _seed_hk_bars(cfg, asof, n_days=90)

    body = build_hk_daily_report(cfg, asof)
    assert "## Section 19 — ETF Flow Analysis" in body
    assert "No HK ETF flow snapshot recorded" in body


def test_hk_stock_connect_block_renders_when_rows_present(tmp_path):
    """D2: §17 surfaces a Southbound subsection (5-session net + last 7 days
    table) when stock_connect_flows has rows."""
    from datetime import timedelta as _td
    cfg = _cfg(tmp_path)
    asof = date(2026, 6, 9)
    _seed_hk_bars(cfg, asof, n_days=90)

    import pandas as pd
    from rotation.ingest.stock_connect_adapter import upsert as sc_upsert
    rows = []
    for i, (d, nb) in enumerate([
        (date(2026, 6, 3), 186.82),
        (date(2026, 6, 4),  -3.15),
        (date(2026, 6, 5), -24.26),
        (date(2026, 6, 8), 113.18),
        (date(2026, 6, 9), -86.14),
    ]):
        rows.append({"ts": d, "net_buy_cny_100m": nb,
                     "hist_cum_cny_100m": 5.4, "holding_value_cny": 1.2e13})
    with connect(cfg.storage.duckdb_path) as con:
        sc_upsert(con, "southbound", pd.DataFrame(rows))

    body = build_hk_daily_report(cfg, asof)
    assert "### Stock Connect — Southbound (Mainland → HK)" in body
    # 5-session sum = 186.82 - 3.15 - 24.26 + 113.18 - 86.14 = 186.45
    assert "+186.4" in body or "+186.5" in body  # tolerate rounding
    # All five sessions present in the table
    for ts in ("2026-06-03", "2026-06-04", "2026-06-05", "2026-06-08", "2026-06-09"):
        assert ts in body


def test_hk_stock_connect_block_handles_missing_table(tmp_path):
    """If stock_connect_flows table doesn't exist (fresh DB), §17 still renders
    without raising — the block falls back to its 'no data yet' placeholder."""
    cfg = _cfg(tmp_path)
    asof = date(2026, 6, 9)
    _seed_hk_bars(cfg, asof, n_days=90)

    body = build_hk_daily_report(cfg, asof)
    assert "### Stock Connect — Southbound (Mainland → HK)" in body
    assert "No Southbound flow data yet" in body
