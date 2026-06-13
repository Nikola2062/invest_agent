"""D2: Stock Connect adapter — schema, upsert, panel, idempotency.

These tests synthesise akshare-shaped frames so the real network call isn't
exercised in CI. The integration with the live Eastmoney endpoint is verified
in the daily pipeline run (any source-side failure surfaces as a `stock_connect`
step error in the run log, never crashes the rest of the pipeline)."""
from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd

from rotation.ingest import stock_connect_adapter as SC


def _akshare_shaped(rows: list[tuple]) -> pd.DataFrame:
    """rows = [(date, net_buy_100m, hist_cum_100m, holding_value), ...]"""
    return pd.DataFrame(
        rows,
        columns=["日期", "当日成交净买额", "历史累计净买额", "持股市值"],
    )


def test_normalises_chinese_columns_and_drops_null_rows(monkeypatch):
    raw = _akshare_shaped([
        (date(2026, 6, 5),  -24.26,  5.3994, 1.22e13),
        (date(2026, 6, 8), +113.18,  5.4107, 1.20e13),
        (date(2026, 6, 9),  None,    None,    None),  # post-cutoff NaN row
    ])
    import akshare as ak  # ensure available before monkeypatching
    monkeypatch.setattr(ak, "stock_hsgt_hist_em", lambda symbol: raw)

    out = SC.fetch_history("southbound")
    assert list(out.columns) == ["ts", "net_buy_cny_100m",
                                 "hist_cum_cny_100m", "holding_value_cny"]
    assert len(out) == 2  # the NaN row is dropped
    assert out["ts"].iloc[-1] == date(2026, 6, 8)


def test_upsert_idempotent_and_overwrites_existing(tmp_path, monkeypatch):
    db = tmp_path / "rot.duckdb"
    con = duckdb.connect(str(db))
    df1 = pd.DataFrame({
        "ts": [date(2026, 6, 5), date(2026, 6, 8)],
        "net_buy_cny_100m": [-24.26, 113.18],
        "hist_cum_cny_100m": [5.3994, 5.4107],
        "holding_value_cny": [1.22e13, 1.20e13],
    })
    n = SC.upsert(con, "southbound", df1)
    assert n == 2
    n_again = SC.upsert(con, "southbound", df1)
    assert n_again == 2  # idempotent on key (ts, direction)
    total = con.execute("SELECT COUNT(*) FROM stock_connect_flows").fetchone()[0]
    assert total == 2

    # Overwrite one row with a corrected value.
    df_fixed = df1.copy()
    df_fixed.loc[df_fixed["ts"] == date(2026, 6, 8), "net_buy_cny_100m"] = 999.99
    SC.upsert(con, "southbound", df_fixed)
    val = con.execute(
        "SELECT net_buy_cny_100m FROM stock_connect_flows "
        "WHERE ts = ? AND direction = ?",
        [date(2026, 6, 8), "southbound"],
    ).fetchone()[0]
    assert val == 999.99


def test_load_panel_returns_directions_as_columns(tmp_path):
    db = tmp_path / "rot.duckdb"
    con = duckdb.connect(str(db))
    rows_sb = pd.DataFrame({
        "ts": [date(2026, 6, 5), date(2026, 6, 8)],
        "net_buy_cny_100m": [-24.26, 113.18],
        "hist_cum_cny_100m": [5.3994, 5.4107],
        "holding_value_cny": [1.22e13, 1.20e13],
    })
    rows_nb = rows_sb.copy()
    rows_nb["net_buy_cny_100m"] = [10.0, -5.0]
    SC.upsert(con, "southbound", rows_sb)
    SC.upsert(con, "northbound", rows_nb)
    panel = SC.load_panel(con, asof=date(2026, 6, 9), lookback_days=90)
    assert set(panel.columns) == {"southbound", "northbound"}
    assert panel.loc[pd.Timestamp("2026-06-08"), "southbound"] == 113.18
    assert panel.loc[pd.Timestamp("2026-06-08"), "northbound"] == -5.0


def test_akshare_unavailable_returns_empty(monkeypatch):
    """If akshare isn't importable, fetch returns empty (never raises)."""
    import sys
    saved = sys.modules.get("akshare")
    sys.modules["akshare"] = None  # force ImportError on `import akshare`
    try:
        out = SC.fetch_history("southbound")
        assert out.empty
    finally:
        if saved is not None:
            sys.modules["akshare"] = saved
        else:
            del sys.modules["akshare"]


def test_akshare_endpoint_failure_returns_empty(monkeypatch):
    """If akshare raises (network, parse, etc.), we log and return empty."""
    import akshare as ak

    def _boom(symbol):
        raise RuntimeError("eastmoney 502")
    monkeypatch.setattr(ak, "stock_hsgt_hist_em", _boom)
    out = SC.fetch_history("southbound")
    assert out.empty


# ============================================================
# D3 — per-stock Southbound holdings
# ============================================================

def _akshare_holdings_shaped(rows: list[tuple]) -> pd.DataFrame:
    """rows = [(date, code, name, close, daily_pct, shares, mv,
                pct, mv_chg_1, mv_chg_5, mv_chg_10), ...]"""
    return pd.DataFrame(rows, columns=[
        "持股日期", "股票代码", "股票简称", "当日收盘价", "当日涨跌幅",
        "持股数量", "持股市值", "持股数量占发行股百分比",
        "持股市值变化-1日", "持股市值变化-5日", "持股市值变化-10日",
    ])


def test_holdings_normalises_and_zfills_codes(monkeypatch):
    raw = _akshare_holdings_shaped([
        (date(2026, 6, 9), "700",  "腾讯",  453.20, 1.51, 100_000, 4.5e10, 5.0, 8.8e9, -2.8e10, 1.2e10),
        (date(2026, 6, 9), "9988", "阿里巴巴", 116.97, -1.44, 50_000, 2.4e10, 4.0, -3.6e9, -2.8e10, -2.9e10),
    ])
    import akshare as ak
    monkeypatch.setattr(ak, "stock_hsgt_stock_statistics_em",
                        lambda symbol, start_date, end_date: raw)

    out = SC.fetch_holdings(date(2026, 6, 1), date(2026, 6, 9))
    assert list(out["symbol"]) == ["00700", "09988"]  # zero-padded to 5 digits
    assert "mv_chg_5d_hkd" in out.columns
    assert out.loc[out["symbol"] == "09988", "mv_chg_5d_hkd"].iloc[0] == -2.8e10


def test_upsert_holdings_idempotent_and_overwrites(tmp_path):
    db = tmp_path / "rot.duckdb"
    con = duckdb.connect(str(db))
    df = pd.DataFrame({
        "ts": [date(2026, 6, 9), date(2026, 6, 9)],
        "symbol": ["00700", "09988"],
        "name": ["腾讯", "阿里巴巴"],
        "close_hkd": [453.20, 116.97],
        "daily_pct": [1.51, -1.44],
        "shares_held": [100_000, 50_000],
        "market_value_hkd": [4.5e10, 2.4e10],
        "pct_of_shares_outstanding": [5.0, 4.0],
        "mv_chg_1d_hkd": [8.8e9, -3.6e9],
        "mv_chg_5d_hkd": [-2.8e10, -2.8e10],
        "mv_chg_10d_hkd": [1.2e10, -2.9e10],
    })
    assert SC.upsert_holdings(con, df) == 2
    # Idempotent
    assert SC.upsert_holdings(con, df) == 2
    total = con.execute("SELECT COUNT(*) FROM stock_connect_holdings").fetchone()[0]
    assert total == 2


def test_load_holdings_for_universe_maps_codes(tmp_path):
    db = tmp_path / "rot.duckdb"
    con = duckdb.connect(str(db))
    SC.upsert_holdings(con, pd.DataFrame({
        "ts": [date(2026, 6, 9)],
        "symbol": ["00700"],
        "name": ["腾讯"],
        "close_hkd": [453.20], "daily_pct": [1.51],
        "shares_held": [100_000], "market_value_hkd": [4.5e10],
        "pct_of_shares_outstanding": [5.0],
        "mv_chg_1d_hkd": [8.8e9], "mv_chg_5d_hkd": [-2.8e10],
        "mv_chg_10d_hkd": [1.2e10],
    }))
    out = SC.load_holdings_for_universe(con, ["0700.HK", "9988.HK"],
                                         asof=date(2026, 6, 9))
    assert len(out) == 1  # only 0700 has data
    assert out["symbol_dot_hk"].iloc[0] == "0700.HK"


def test_load_holdings_returns_empty_when_table_missing(tmp_path):
    """Fresh DB without the holdings table — must not raise."""
    db = tmp_path / "rot.duckdb"
    con = duckdb.connect(str(db))
    out = SC.load_holdings_for_universe(con, ["0700.HK"], asof=date(2026, 6, 9))
    assert out.empty


def test_hk_code_5digit_normalises():
    assert SC._hk_code_5digit("0700.HK") == "00700"
    assert SC._hk_code_5digit("700.HK") == "00700"
    assert SC._hk_code_5digit("9988.HK") == "09988"
    assert SC._hk_code_5digit("00388.HK") == "00388"
