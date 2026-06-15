"""Structured, multi-horizon outcomes store for calibrated reflection.

The existing ``TradingMemoryLog`` writes a human-readable markdown log,
one 5-day outcome per trade. That's good for audit but bad for
statistics — you can't compute hit rates across hundreds of trades by
regex'ing markdown.

This store is the *companion* to the markdown log: an append-only CSV
that stores the same trades with **multi-horizon** outcomes (5d, 21d,
63d alpha) plus enough metadata (rating, sector) to bucket and
significance-test. The markdown log stays the source of truth for the
agent prompt's per-ticker context; this store powers the aggregated
calibrated-lessons block.

Append-only: each new outcome appends a row. A given (ticker, date) is
written exactly once. Idempotent re-runs are safe (the writer checks
before append).
"""

from __future__ import annotations

import csv
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)


# Default horizons (trading days). 5d catches reactive noise, 21d ≈ 1 month
# is the standard equity research horizon, 63d ≈ 1 quarter captures slower
# theses. The aggregator computes stats at all three so the PM prompt can
# surface whichever shows signal.
DEFAULT_HORIZONS: tuple[int, ...] = (5, 21, 63)


@dataclass
class OutcomeRow:
    """One trade outcome, sized for statistical aggregation."""

    ticker: str
    trade_date: str       # YYYY-MM-DD
    rating: str           # one of RATINGS_5_TIER
    sector: str = "Unknown"
    benchmark: str = "SPY"
    # Per-horizon raw and alpha returns. None when the horizon isn't yet
    # realisable (e.g. 63d outcome 30 days after trade). Re-resolved on
    # subsequent passes once enough trading days have elapsed.
    horizons: dict[int, dict[str, Optional[float]]] = field(default_factory=dict)

    def flatten(self) -> dict[str, str | float | None]:
        """Flatten to a CSV-friendly row keyed by stable column names."""
        out: dict[str, str | float | None] = {
            "ticker": self.ticker,
            "trade_date": self.trade_date,
            "rating": self.rating,
            "sector": self.sector,
            "benchmark": self.benchmark,
        }
        for h, vals in self.horizons.items():
            out[f"raw_{h}d"] = vals.get("raw")
            out[f"alpha_{h}d"] = vals.get("alpha")
        return out


class OutcomesStore:
    """Append-only CSV-backed store of multi-horizon trade outcomes."""

    def __init__(self, path: str | Path, horizons: Iterable[int] = DEFAULT_HORIZONS):
        self._path = Path(path).expanduser()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._horizons = tuple(sorted(horizons))
        self._columns = ["ticker", "trade_date", "rating", "sector", "benchmark"]
        for h in self._horizons:
            self._columns.extend([f"raw_{h}d", f"alpha_{h}d"])

    @property
    def path(self) -> Path:
        return self._path

    @property
    def horizons(self) -> tuple[int, ...]:
        return self._horizons

    def append(self, row: OutcomeRow) -> bool:
        """Add a row if (ticker, trade_date) is not already present.

        Returns True if a row was written, False if it was a duplicate.
        Uses a single-pass scan rather than loading the full table to keep
        this cheap on long histories. Atomic at the row level — partial
        writes can't happen because csv.writer flushes on close.
        """
        if self._has_pair(row.ticker, row.trade_date):
            return False

        new_file = not self._path.exists()
        with self._path.open("a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._columns)
            if new_file:
                writer.writeheader()
            flat = row.flatten()
            writer.writerow({k: flat.get(k) for k in self._columns})
        return True

    def update(self, row: OutcomeRow) -> bool:
        """Replace an existing (ticker, trade_date) row with new horizon values.

        Useful when a later run can resolve a longer horizon that was None
        at first write. Falls back to ``append`` when no existing row matches.
        Performs an atomic temp-file rewrite.
        """
        if not self._path.exists():
            return self.append(row)
        rows = list(self._read_all())
        idx = next(
            (i for i, r in enumerate(rows)
             if r.get("ticker") == row.ticker and r.get("trade_date") == row.trade_date),
            None,
        )
        if idx is None:
            return self.append(row)
        rows[idx] = {k: row.flatten().get(k) for k in self._columns}
        self._write_all(rows)
        return True

    def load(self) -> list[dict]:
        """Read every row as a dict. Returns [] when the file doesn't exist."""
        if not self._path.exists():
            return []
        return list(self._read_all())

    # --- internals ---

    def _has_pair(self, ticker: str, trade_date: str) -> bool:
        if not self._path.exists():
            return False
        with self._path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("ticker") == ticker and r.get("trade_date") == trade_date:
                    return True
        return False

    def _read_all(self):
        with self._path.open("r", encoding="utf-8", newline="") as f:
            yield from csv.DictReader(f)

    def _write_all(self, rows: list[dict]) -> None:
        tmp = self._path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._columns)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: r.get(k) for k in self._columns})
        os.replace(tmp, self._path)
