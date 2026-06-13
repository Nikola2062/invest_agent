"""Shared fetch → validate → quarantine → upsert runner.

Previously duplicated in cli._run_for_date and pipeline._do_fetch; this is the
single source of truth for ingest-day semantics, including the §1.7 coverage
gate: a run whose NYSE-aligned coverage falls below `ingest.min_coverage_pct`
is logged as `degraded` and flagged so the caller can skip signal publication.
"""
from __future__ import annotations

import logging
from datetime import date

import duckdb

from ..config import Config
from ..store import log_run_finish, log_run_start, quarantine, upsert_bars
from .validator import is_stale, validate_row
from .yfinance_adapter import fetch_bars

log = logging.getLogger(__name__)

# HK-listed symbols trade on HKEX and are excluded from the NYSE-aligned
# signals panel (the project docs §1.6.1 decision 4), so they don't count toward the
# coverage gate — an HKEX holiday must not degrade a clean NYSE run.
_COVERAGE_EXEMPT_CLASSES = {"equity_hk"}


def fetch_validate_store(
    cfg: Config,
    con: duckdb.DuckDBPyConnection,
    run_id: str,
    asof: date,
) -> dict:
    log_run_start(con, run_id, asof, n_symbols=len(cfg.universe))

    # C3: any failure after log_run_start must finish the run_log row, or it is
    # orphaned in status='running'. The whole body is guarded so a fault in
    # validation / upsert / coverage (not just fetch) records a terminal state.
    try:
        raw = fetch_bars(cfg, asof)

        valid: list[dict] = []
        n_quarantined = 0
        for r in raw:
            threshold = cfg.ingest.outlier_threshold(r["asset_class"])
            ok, reason = validate_row(r, threshold)
            if not ok:
                quarantine(con, r, reason)
                n_quarantined += 1
                log.warning("quarantine %s %s: %s", r["symbol"], r["ts"], reason)
                continue
            if is_stale(r["ts"], asof, cfg.ingest.stale_threshold_days):
                r["stale"] = True
            valid.append(r)

        result = upsert_bars(con, valid)

        gated = {s.symbol for s in cfg.universe
                 if s.asset_class not in _COVERAGE_EXEMPT_CLASSES}
        # Coverage counts only bars AT the asof date. The fetch window spans ~10
        # days, so counting any-row-in-window would mask exactly the failure this
        # gate exists for: the provider returning nothing (or NaN closes) for the
        # most recent session while the lookback days are fine.
        got = {r["symbol"] for r in valid if r["ts"] == asof}
        coverage = len(got & gated) / max(len(gated), 1)
        coverage_ok = coverage * 100.0 >= cfg.ingest.min_coverage_pct
        notes = "" if coverage_ok else (
            f"coverage {coverage:.0%} of NYSE-aligned universe below "
            f"{cfg.ingest.min_coverage_pct:.0f}% gate (design §1.7)"
        )
        log_run_finish(
            con, run_id, "ok" if coverage_ok else "degraded",
            n_inserted=result.inserted, n_updated=result.updated,
            n_quarantined=n_quarantined, notes=notes,
        )
    except Exception as exc:
        log_run_finish(con, run_id, "failed", 0, 0, 0, f"{type(exc).__name__}: {exc}")
        raise

    if not coverage_ok:
        log.warning("run %s degraded: %s", run_id, notes)

    return {
        "run_id": run_id,
        "asof": asof.isoformat(),
        "fetched": len(raw),
        "inserted": result.inserted,
        "updated": result.updated,
        "quarantined": n_quarantined,
        "coverage": round(coverage, 4),
        "coverage_ok": coverage_ok,
    }
