"""D4: China macro adapter — RRR, LPR, SHIBOR via akshare.

Synthetic akshare-shaped frames; the real endpoint is verified by running the
daily pipeline. Tests focus on column normalisation, dedup of same-day RRR
rows (GFC-era data quirk), upsert idempotency, and the `latest_value`
helper used by the report."""
from __future__ import annotations

from datetime import date

import duckdb
import pandas as pd

from rotation.ingest import china_macro_adapter as CM


def _rrr_raw(rows: list[tuple]) -> pd.DataFrame:
    """rows = [('YYYY年MM月DD日', large_after, magnitude), ...]"""
    return pd.DataFrame(rows, columns=[
        "公布时间", "大型金融机构-调整后", "大型金融机构-调整幅度",
    ])


def _lpr_raw(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["TRADE_DATE", "LPR1Y", "LPR5Y", "RATE_1", "RATE_2"])


def _shibor_raw(rows: list[tuple]) -> pd.DataFrame:
    cols = ["日期"] + [f"{t}-{f}" for t in ("O/N","1W","2W","1M","3M","6M","9M","1Y")
                       for f in ("定价", "涨跌幅")]
    return pd.DataFrame(rows, columns=cols)


def test_rrr_drops_same_day_duplicates(monkeypatch):
    """The Eastmoney source can return two rows for the same announce date
    (e.g. 2008-06-07). We must dedup — otherwise upsert hits a PK violation."""
    raw = _rrr_raw([
        ("2008年06月07日", 17.00, 0.50),
        ("2008年06月07日", 17.50, 1.00),  # second tranche, same announce date
        ("2025年05月07日",  9.00, -0.50),
    ])
    import akshare as ak
    monkeypatch.setattr(ak, "macro_china_reserve_requirement_ratio", lambda: raw)
    out = CM.fetch_rrr()
    assert out["ts"].is_unique
    assert len(out) == 2  # two distinct announce dates


def test_lpr_normalises_columns(monkeypatch):
    raw = _lpr_raw([
        ("2026-05-20", 3.0, 3.5, 4.35, 4.9),
        ("2026-04-20", 3.0, 3.5, 4.35, 4.9),
    ])
    import akshare as ak
    monkeypatch.setattr(ak, "macro_china_lpr", lambda: raw)
    out = CM.fetch_lpr()
    assert list(out.columns) == ["ts", "lpr_1y", "lpr_5y"]
    assert (out["lpr_1y"] == 3.0).all()


def test_shibor_keeps_3m_and_1y(monkeypatch):
    raw = _shibor_raw([
        ("2026-06-09", 1.384, 3.0, 1.420, 1.9, 1.402, 0.7, 1.391, 0.25,
                       1.402, 0.0, 1.4275, 0.1, 1.450, 0.3, 1.460, 0.0),
    ])
    import akshare as ak
    monkeypatch.setattr(ak, "macro_china_shibor_all", lambda: raw)
    out = CM.fetch_shibor()
    assert "shibor_3m" in out.columns and "shibor_1y" in out.columns
    assert out["shibor_3m"].iloc[0] == 1.402
    assert out["shibor_1y"].iloc[0] == 1.460


def test_upsert_idempotent_and_overwrites(tmp_path):
    db = tmp_path / "rot.duckdb"
    con = duckdb.connect(str(db))
    rows = [
        {"ts": date(2026, 5, 20), "series_id": "LPR_1Y", "value": 3.0},
        {"ts": date(2026, 5, 20), "series_id": "LPR_5Y", "value": 3.5},
    ]
    assert CM._upsert_long(con, rows) == 2
    # Idempotent re-run
    assert CM._upsert_long(con, rows) == 2
    total = con.execute("SELECT COUNT(*) FROM china_macro_series").fetchone()[0]
    assert total == 2


def test_latest_value_reads_meta_json(tmp_path):
    db = tmp_path / "rot.duckdb"
    con = duckdb.connect(str(db))
    import json
    CM._upsert_long(con, [{
        "ts": date(2025, 5, 7), "series_id": "RRR_LARGE_BANKS",
        "value": 9.0, "meta_json": json.dumps({"magnitude": -0.5}),
    }])
    v = CM.latest_value(con, "RRR_LARGE_BANKS", asof=date(2026, 6, 9))
    assert v == {"ts": date(2025, 5, 7), "value": 9.0, "magnitude": -0.5}


def test_latest_value_table_missing(tmp_path):
    db = tmp_path / "rot.duckdb"
    con = duckdb.connect(str(db))
    assert CM.latest_value(con, "RRR_LARGE_BANKS") is None


def test_akshare_endpoint_failure_returns_empty(monkeypatch):
    import akshare as ak

    def _boom():
        raise RuntimeError("eastmoney 502")
    monkeypatch.setattr(ak, "macro_china_reserve_requirement_ratio", _boom)
    monkeypatch.setattr(ak, "macro_china_lpr", _boom)
    monkeypatch.setattr(ak, "macro_china_shibor_all", _boom)
    assert CM.fetch_rrr().empty
    assert CM.fetch_lpr().empty
    assert CM.fetch_shibor().empty
