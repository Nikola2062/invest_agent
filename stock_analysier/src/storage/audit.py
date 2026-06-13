"""SQLite audit trail for analysis runs and thesis-break alerts.

Every AnalysisResult is persisted (full JSON + denormalized scalar columns for
querying). The Monitor agent reads recent runs to detect thesis breaks. The
dashboard can render history. Post-mortems become possible.

Schema is created lazily on first call. Storage path: data/audit.sqlite.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from src.models.schemas import AnalysisResult

log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "audit.sqlite"


SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    market TEXT NOT NULL,
    timestamp_utc TEXT NOT NULL,
    current_price REAL NOT NULL,
    currency TEXT NOT NULL,
    quality_score REAL,
    moat_strength TEXT,
    intrinsic_low REAL,
    intrinsic_base REAL,
    intrinsic_high REAL,
    margin_of_safety_pct REAL,
    valuation_confidence TEXT,
    realized_vol_pct REAL,
    p_dd_10 REAL,
    p_dd_15 REAL,
    p_dd_20 REAL,
    p_dd_25 REAL,
    tactical_level INTEGER,
    tactical_label TEXT,
    if_held_action TEXT,
    if_not_held_recommendation TEXT,
    sentiment_score REAL,
    devil_verdict TEXT,
    devil_summary TEXT,
    composite_tech_signal TEXT,
    key_support REAL,
    full_result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_runs_symbol_ts ON analysis_runs(symbol, timestamp_utc DESC);

CREATE TABLE IF NOT EXISTS thesis_break_alerts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    detected_at_utc TEXT NOT NULL,
    severity TEXT NOT NULL,
    category TEXT NOT NULL,
    summary TEXT NOT NULL,
    evidence_json TEXT NOT NULL,
    acknowledged_at_utc TEXT,
    pushed_to_telegram INTEGER DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_alerts_symbol ON thesis_break_alerts(symbol, detected_at_utc DESC);
"""


@contextmanager
def _conn(db_path: Path = DEFAULT_DB_PATH) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, isolation_level=None)  # autocommit
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript(SCHEMA)
        _migrate(conn)
        yield conn
    finally:
        conn.close()


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after the original schema. Safe to run on every open."""
    cols = {row[1] for row in conn.execute("PRAGMA table_info(analysis_runs)")}
    if "composite_tech_signal" not in cols:
        conn.execute("ALTER TABLE analysis_runs ADD COLUMN composite_tech_signal TEXT")
    if "key_support" not in cols:
        conn.execute("ALTER TABLE analysis_runs ADD COLUMN key_support REAL")


def record_analysis(result: AnalysisResult, db_path: Path = DEFAULT_DB_PATH) -> int:
    """Persist an AnalysisResult. Returns the new row id."""
    payload = result.model_dump_json()
    p_dd = result.risk.drawdown_probabilities
    da_verdict = None
    da_summary = None
    # Devil's Advocate is attached separately; check for the attr
    devil = getattr(result, "devil_advocate", None)
    if devil is not None:
        da_verdict = devil.overall_verdict
        da_summary = devil.summary[:500] if devil.summary else None

    tech_signal = result.technical.composite_signal if result.technical else None
    key_support = (
        result.technical.price_map.key_support
        if result.technical and result.technical.price_map
        else None
    )

    with _conn(db_path) as c:
        cur = c.execute(
            """
            INSERT INTO analysis_runs (
              symbol, market, timestamp_utc, current_price, currency,
              quality_score, moat_strength,
              intrinsic_low, intrinsic_base, intrinsic_high, margin_of_safety_pct, valuation_confidence,
              realized_vol_pct, p_dd_10, p_dd_15, p_dd_20, p_dd_25,
              tactical_level, tactical_label, if_held_action, if_not_held_recommendation,
              sentiment_score, devil_verdict, devil_summary,
              composite_tech_signal, key_support,
              full_result_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                result.symbol, result.market, result.timestamp_utc.isoformat(),
                result.current_price, result.currency,
                result.fundamental.quality_score, result.fundamental.moat_strength,
                result.valuation.intrinsic_low, result.valuation.intrinsic_base,
                result.valuation.intrinsic_high, result.valuation.margin_of_safety_pct,
                result.valuation.confidence,
                result.risk.realized_vol_annualized_pct,
                float(p_dd.get("10", 0)), float(p_dd.get("15", 0)),
                float(p_dd.get("20", 0)), float(p_dd.get("25", 0)),
                result.if_held.tactical.level, result.if_held.tactical.label,
                result.if_held.tactical.action,
                result.if_not_held.recommendation,
                result.forward_catalysts.sentiment_score if result.forward_catalysts else None,
                da_verdict, da_summary,
                tech_signal, key_support,
                payload,
            ),
        )
        return cur.lastrowid


def get_recent_runs(symbol: str, limit: int = 10, db_path: Path = DEFAULT_DB_PATH) -> list[sqlite3.Row]:
    with _conn(db_path) as c:
        cur = c.execute(
            "SELECT * FROM analysis_runs WHERE symbol = ? ORDER BY timestamp_utc DESC LIMIT ?",
            (symbol, limit),
        )
        return list(cur.fetchall())


def get_latest_run(symbol: str, db_path: Path = DEFAULT_DB_PATH) -> Optional[sqlite3.Row]:
    runs = get_recent_runs(symbol, limit=1, db_path=db_path)
    return runs[0] if runs else None


def get_previous_run_before(symbol: str, ts: datetime, db_path: Path = DEFAULT_DB_PATH) -> Optional[sqlite3.Row]:
    """Get the most recent run STRICTLY BEFORE the given timestamp."""
    with _conn(db_path) as c:
        cur = c.execute(
            "SELECT * FROM analysis_runs WHERE symbol = ? AND timestamp_utc < ? ORDER BY timestamp_utc DESC LIMIT 1",
            (symbol, ts.isoformat()),
        )
        row = cur.fetchone()
        return row


def record_alert(
    symbol: str,
    severity: str,
    category: str,
    summary: str,
    evidence: dict,
    db_path: Path = DEFAULT_DB_PATH,
) -> int:
    with _conn(db_path) as c:
        cur = c.execute(
            """
            INSERT INTO thesis_break_alerts (
              symbol, detected_at_utc, severity, category, summary, evidence_json
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                symbol,
                datetime.now(timezone.utc).isoformat(),
                severity,
                category,
                summary,
                json.dumps(evidence, default=str),
            ),
        )
        return cur.lastrowid


def mark_alert_pushed(alert_id: int, db_path: Path = DEFAULT_DB_PATH) -> None:
    with _conn(db_path) as c:
        c.execute("UPDATE thesis_break_alerts SET pushed_to_telegram = 1 WHERE id = ?", (alert_id,))


def get_unpushed_alerts(db_path: Path = DEFAULT_DB_PATH) -> list[sqlite3.Row]:
    with _conn(db_path) as c:
        cur = c.execute(
            "SELECT * FROM thesis_break_alerts WHERE pushed_to_telegram = 0 ORDER BY detected_at_utc"
        )
        return list(cur.fetchall())


def get_recent_alerts(symbol: Optional[str] = None, limit: int = 20, db_path: Path = DEFAULT_DB_PATH) -> list[sqlite3.Row]:
    with _conn(db_path) as c:
        if symbol:
            cur = c.execute(
                "SELECT * FROM thesis_break_alerts WHERE symbol = ? ORDER BY detected_at_utc DESC LIMIT ?",
                (symbol, limit),
            )
        else:
            cur = c.execute(
                "SELECT * FROM thesis_break_alerts ORDER BY detected_at_utc DESC LIMIT ?",
                (limit,),
            )
        return list(cur.fetchall())


def count_recent_runs_with_signal(
    symbol: str, days: int, signal_predicate_sql: str, db_path: Path = DEFAULT_DB_PATH
) -> int:
    """Count runs in the last N days where signal_predicate_sql (e.g. 'p_dd_15 > 0.5') is true.

    Used by the (future) persistence tracker.
    """
    with _conn(db_path) as c:
        cur = c.execute(
            f"SELECT COUNT(*) AS n FROM analysis_runs "
            f"WHERE symbol = ? AND timestamp_utc >= datetime('now', ?) "
            f"AND ({signal_predicate_sql})",
            (symbol, f"-{days} days"),
        )
        return int(cur.fetchone()["n"])


def prune_old_runs(
    days_to_keep: int = 365,
    keep_min_per_symbol: int = 30,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, int]:
    """Delete audit rows older than `days_to_keep` while keeping the most-recent
    `keep_min_per_symbol` rows per symbol regardless of age.

    Without this, the audit DB grows unboundedly: roughly
    (N watchlist tickers) × (4 daily pushes) = ~20 rows/day, ~7300/year.
    Post-mortem analysis benefits from history but doesn't need 5-year-old runs.

    Returns a dict {runs_deleted, alerts_deleted} so the caller can log it.
    Safe to call regularly (e.g. weekly cron). The `keep_min_per_symbol` floor
    protects fresh installs and rarely-analyzed symbols from getting truncated
    below the cold-start history threshold.
    """
    cutoff_clause = f"-{int(days_to_keep)} days"
    with _conn(db_path) as c:
        # Delete rows that are BOTH older than cutoff AND outside the per-symbol
        # most-recent-N protected set.
        cur = c.execute(
            """
            DELETE FROM analysis_runs WHERE id NOT IN (
              SELECT id FROM analysis_runs
              WHERE timestamp_utc >= datetime('now', ?)
              UNION
              SELECT id FROM (
                SELECT id,
                       ROW_NUMBER() OVER (PARTITION BY symbol ORDER BY timestamp_utc DESC) AS rn
                FROM analysis_runs
              )
              WHERE rn <= ?
            )
            """,
            (cutoff_clause, keep_min_per_symbol),
        )
        runs_deleted = cur.rowcount or 0

        cur = c.execute(
            "DELETE FROM thesis_break_alerts WHERE detected_at_utc < datetime('now', ?)",
            (cutoff_clause,),
        )
        alerts_deleted = cur.rowcount or 0

    log.info(
        "Audit prune: removed %d run(s) older than %dd (keeping ≥%d per symbol), "
        "and %d expired alert(s).",
        runs_deleted, days_to_keep, keep_min_per_symbol, alerts_deleted,
    )
    return {"runs_deleted": runs_deleted, "alerts_deleted": alerts_deleted}


if __name__ == "__main__":
    # CLI: prune the audit DB. Default keeps 365 days + at least 30 runs per symbol.
    import argparse
    parser = argparse.ArgumentParser(description="Prune old audit rows.")
    parser.add_argument("--days", type=int, default=365, help="Keep rows newer than N days")
    parser.add_argument("--min-per-symbol", type=int, default=30,
                        help="Always keep at least N most-recent runs per symbol")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    out = prune_old_runs(days_to_keep=args.days, keep_min_per_symbol=args.min_per_symbol)
    print(f"Pruned: {out['runs_deleted']} run(s), {out['alerts_deleted']} alert(s).")
