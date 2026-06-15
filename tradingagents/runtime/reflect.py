"""Phase 4 — close the loop: log every overlay decision for later scoring.

Each digest records its per-name ratings into the existing append-only
``OutcomesStore``. The standard machinery (return back-fill + ``reflection_v2``
aggregation) then grades those calls against realised forward returns — a process
you don't grade is noise. Append-only and idempotent per (ticker, date).
"""

from __future__ import annotations

from pathlib import Path

from tradingagents.portfolio.outcomes_store import OutcomesStore, OutcomeRow


def record_overlay_outcomes(verdicts: list[dict], date: str, path: str | Path) -> int:
    """Append one OutcomeRow per verdict (TA rating). Returns rows newly written."""
    store = OutcomesStore(path)
    written = 0
    for v in verdicts:
        row = OutcomeRow(ticker=v["symbol"], trade_date=date, rating=v["rating"])
        if store.append(row):
            written += 1
    return written
