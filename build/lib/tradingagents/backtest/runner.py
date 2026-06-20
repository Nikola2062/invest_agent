"""Walk-forward runner — iterates (ticker, date) pairs through a Strategy.

Design choices:

  - **Append-only JSONL output.** One decision per line. If the process
    crashes (network, OOM, ^C) the next invocation reads what was already
    written and skips those pairs. No checkpoint files, no DB.

  - **Date generation is explicit.** The caller passes a list of rebalance
    dates (or uses ``generate_rebalance_dates``). The runner never invents
    its own calendar — bad calendar logic is the #1 source of subtle
    backtest bugs.

  - **Decisions emit in (date, ticker) order**, not (ticker, date), so a
    partial run still represents a coherent slice of time. This matters
    because the portfolio module needs all tickers' decisions at a given
    rebalance to compute weights.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence

from .strategy import Decision, Strategy

logger = logging.getLogger(__name__)


# --- date helpers ---------------------------------------------------------


def generate_rebalance_dates(
    start: str,
    end: str,
    freq: str = "monthly",
) -> list[str]:
    """Yield rebalance dates between ``start`` and ``end`` inclusive.

    ``freq`` is one of ``"daily"``, ``"weekly"``, ``"monthly"``, ``"quarterly"``.
    Dates are returned as YYYY-MM-DD strings — the rest of the harness uses
    string dates throughout so JSONL round-trips cleanly without coercion.

    No trading-calendar awareness: a Saturday rebalance is fine because the
    portfolio module looks up the *next* trading day's close. Adding holiday
    awareness here would couple the runner to an exchange.
    """
    start_d = datetime.strptime(start, "%Y-%m-%d").date()
    end_d = datetime.strptime(end, "%Y-%m-%d").date()
    if end_d < start_d:
        raise ValueError(f"end {end!r} is before start {start!r}")

    step = {
        "daily": timedelta(days=1),
        "weekly": timedelta(days=7),
        "monthly": None,  # handled via month arithmetic below
        "quarterly": None,
    }.get(freq.lower())

    out: list[str] = []
    cur = start_d
    while cur <= end_d:
        out.append(cur.strftime("%Y-%m-%d"))
        if freq.lower() == "monthly":
            cur = _add_months(cur, 1)
        elif freq.lower() == "quarterly":
            cur = _add_months(cur, 3)
        else:
            if step is None:
                raise ValueError(f"unknown freq {freq!r}")
            cur = cur + step
    return out


def _add_months(d: date, n: int) -> date:
    """Add n calendar months, clamping day-of-month to month length."""
    month = d.month - 1 + n
    year = d.year + month // 12
    month = month % 12 + 1
    # Last valid day in target month
    if month == 12:
        last = 31
    else:
        last = (date(year, month + 1, 1) - timedelta(days=1)).day
    return date(year, month, min(d.day, last))


# --- run state ------------------------------------------------------------


def _load_completed(jsonl_path: Path) -> set[tuple[str, str]]:
    """Return the set of (date, ticker) pairs already recorded in ``jsonl_path``.

    Tolerates malformed lines (crashes mid-write) by skipping them — a
    re-run will overwrite the slot with a fresh decision.
    """
    if not jsonl_path.exists():
        return set()
    completed: set[tuple[str, str]] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                completed.add((row["trade_date"], row["ticker"]))
            except (json.JSONDecodeError, KeyError):
                continue
    return completed


def _append_decision(jsonl_path: Path, decision: Decision) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(asdict(decision), default=str) + "\n")


# --- main entry point -----------------------------------------------------


def walk_forward(
    tickers: Sequence[str],
    dates: Sequence[str],
    strategy: Strategy,
    output_path: str | Path,
    *,
    on_error: str = "skip",
    progress: Optional[Callable[[Decision], None]] = None,
) -> Iterator[Decision]:
    """Run ``strategy`` over every (date, ticker) pair, recording to JSONL.

    Yields each new Decision as it is produced so callers can stream
    progress or build the equity curve incrementally. Re-running over the
    same ``output_path`` resumes after the last completed pair.

    ``on_error``:
      - ``"skip"``: log and continue on strategy failures (recommended for
        long runs — one bad ticker shouldn't kill 500 others).
      - ``"raise"``: propagate exceptions (recommended for tests).
    """
    output_path = Path(output_path)
    completed = _load_completed(output_path)
    total = len(tickers) * len(dates)
    done = len(completed)

    if completed:
        logger.info("Resuming run: %d/%d pairs already recorded", done, total)

    for d in dates:
        for ticker in tickers:
            if (d, ticker) in completed:
                continue
            t0 = time.monotonic()
            try:
                decision = strategy(ticker, d)
            except Exception as exc:
                if on_error == "raise":
                    raise
                logger.warning(
                    "Strategy failed for %s on %s: %s — skipping", ticker, d, exc,
                )
                continue
            decision.runtime_seconds = decision.runtime_seconds or (
                time.monotonic() - t0
            )
            _append_decision(output_path, decision)
            done += 1
            if progress is not None:
                progress(decision)
            yield decision


def load_decisions(jsonl_path: str | Path) -> list[Decision]:
    """Read decisions back from a JSONL file produced by ``walk_forward``."""
    p = Path(jsonl_path)
    if not p.exists():
        return []
    out: list[Decision] = []
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            out.append(Decision(
                ticker=row["ticker"],
                trade_date=row["trade_date"],
                rating=row["rating"],
                raw_decision=row.get("raw_decision", ""),
                state_log_path=row.get("state_log_path", ""),
                runtime_seconds=float(row.get("runtime_seconds", 0.0)),
                extra=row.get("extra", {}),
            ))
    return out
