from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Iterable, Iterator

import duckdb
import pandas as pd

from .schema import ensure_schema


@dataclass
class UpsertResult:
    inserted: int
    updated: int  # rows where existing OHLCV differed and revision was bumped


@contextmanager
def connect(db_path: Path) -> Iterator[duckdb.DuckDBPyConnection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        ensure_schema(con)
        yield con
    finally:
        con.close()


_COMPARE_COLS = ("open", "high", "low", "close", "adj_close", "volume")


def upsert_bars(
    con: duckdb.DuckDBPyConnection,
    bars: Iterable[dict],
) -> UpsertResult:
    """Idempotent upsert. Same (symbol, ts, OHLCV) -> no-op. Different OHLCV -> revision++.

    Returns counts of inserted and revision-bumped rows. A re-run with identical data
    yields (0, 0) — this is what gives us deterministic replay.
    """
    rows = list(bars)
    if not rows:
        return UpsertResult(0, 0)

    df = pd.DataFrame(rows)
    expected = {
        "symbol", "asset_class", "ts", "open", "high", "low", "close",
        "adj_close", "volume", "source", "ingested_at", "stale",
    }
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"upsert_bars: missing columns {missing}")

    # Normalize ts to datetime64 — DuckDB returns DATE columns as datetime64[us],
    # so the join key on both sides must match.
    df["ts"] = pd.to_datetime(df["ts"])

    con.register("incoming", df)
    try:
        existing = con.execute(
            """
            SELECT i.symbol, i.ts,
                   r.open AS r_open, r.high AS r_high, r.low AS r_low,
                   r.close AS r_close, r.adj_close AS r_adj_close, r.volume AS r_volume,
                   r.revision AS r_revision
            FROM incoming i
            LEFT JOIN raw_bars r USING (symbol, ts)
            """
        ).df()

        merged = df.merge(existing, on=["symbol", "ts"], how="left")
        merged["is_new"] = merged["r_revision"].isna()

        def _differs(row: pd.Series) -> bool:
            if row["is_new"]:
                return False
            for col in _COMPARE_COLS:
                a, b = row[col], row[f"r_{col}"]
                if pd.isna(a) and pd.isna(b):
                    continue
                if pd.isna(a) or pd.isna(b):
                    return True
                if abs(float(a) - float(b)) > 1e-9:
                    return True
            return False

        merged["differs"] = merged.apply(_differs, axis=1)
        merged["revision"] = merged.apply(
            lambda r: 0 if r["is_new"] else (int(r["r_revision"]) + 1 if r["differs"] else int(r["r_revision"])),
            axis=1,
        )

        to_write = merged[merged["is_new"] | merged["differs"]].copy()
        if to_write.empty:
            return UpsertResult(0, 0)

        write_cols = [
            "symbol", "asset_class", "ts", "open", "high", "low", "close",
            "adj_close", "volume", "source", "revision", "ingested_at", "stale",
        ]
        to_write = to_write[write_cols]
        con.register("to_write", to_write)
        # DELETE+INSERT must be atomic: DuckDB auto-commits each statement, so a
        # crash between the two would drop rows and break replay determinism.
        con.execute("BEGIN TRANSACTION")
        try:
            con.execute(
                """
                DELETE FROM raw_bars
                WHERE EXISTS (
                    SELECT 1 FROM to_write tw
                    WHERE tw.symbol = raw_bars.symbol AND tw.ts = raw_bars.ts
                )
                """
            )
            con.execute("INSERT INTO raw_bars SELECT * FROM to_write")
            con.execute("COMMIT")
        except Exception:
            con.execute("ROLLBACK")
            raise
        finally:
            con.unregister("to_write")

        inserted = int(merged["is_new"].sum())
        updated = int(merged["differs"].sum())
        return UpsertResult(inserted=inserted, updated=updated)
    finally:
        con.unregister("incoming")


def get_bars(
    con: duckdb.DuckDBPyConnection,
    symbol: str | None = None,
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    clauses, params = [], []
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol)
    if start:
        clauses.append("ts >= ?")
        params.append(start)
    if end:
        clauses.append("ts <= ?")
        params.append(end)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    return con.execute(
        f"SELECT * FROM raw_bars {where} ORDER BY symbol, ts", params
    ).df()


def quarantine(
    con: duckdb.DuckDBPyConnection,
    row: dict,
    reason: str,
) -> None:
    con.execute(
        """
        INSERT INTO raw_bars_quarantine
        (symbol, asset_class, ts, open, high, low, close, adj_close, volume,
         source, reason, quarantined_at)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
        """,
        [
            row["symbol"], row["asset_class"], row["ts"],
            row.get("open"), row.get("high"), row.get("low"),
            row.get("close"), row.get("adj_close"), row.get("volume"),
            row["source"], reason, datetime.utcnow(),
        ],
    )


def load_signals_and_regime(
    con: duckdb.DuckDBPyConnection,
    asof: date,
) -> tuple[dict, dict | None, dict | None]:
    """Load the day's signal scores plus today's and the previous regime rows,
    shaped for alerts.evaluate_triggers. Shared by pipeline and CLI."""
    sig_rows = con.execute(
        "SELECT signal_name, score, confidence FROM signals_daily WHERE ts = ?", [asof]
    ).df()
    signals = {
        r["signal_name"]: {
            "score": None if pd.isna(r["score"]) else float(r["score"]),
            "confidence": None if pd.isna(r["confidence"]) else float(r["confidence"]),
        } for _, r in sig_rows.iterrows()
    }
    reg = con.execute(
        "SELECT regime, prev_regime, confidence FROM regime_history WHERE ts = ?", [asof]
    ).df()
    today_regime = None if reg.empty else {
        "regime": reg.iloc[0]["regime"],
        "prev_regime": reg.iloc[0]["prev_regime"],
        "confidence": float(reg.iloc[0]["confidence"]) if reg.iloc[0]["confidence"] is not None else None,
    }
    prev = con.execute(
        "SELECT regime FROM regime_history WHERE ts < ? ORDER BY ts DESC LIMIT 1", [asof]
    ).df()
    prev_regime = None if prev.empty else {"regime": prev.iloc[0]["regime"]}
    return signals, today_regime, prev_regime


def log_run_start(
    con: duckdb.DuckDBPyConnection,
    run_id: str, asof: date, n_symbols: int,
) -> None:
    con.execute(
        "INSERT INTO run_log (run_id, asof_date, started_at, status, n_symbols) "
        "VALUES (?, ?, ?, 'running', ?)",
        [run_id, asof, datetime.utcnow(), n_symbols],
    )


def log_run_finish(
    con: duckdb.DuckDBPyConnection,
    run_id: str, status: str,
    n_inserted: int, n_updated: int, n_quarantined: int,
    notes: str = "",
) -> None:
    con.execute(
        """
        UPDATE run_log
        SET finished_at = ?, status = ?, n_inserted = ?, n_updated = ?,
            n_quarantined = ?, notes = ?
        WHERE run_id = ?
        """,
        [datetime.utcnow(), status, n_inserted, n_updated, n_quarantined, notes, run_id],
    )
