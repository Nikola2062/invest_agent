from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Symbol:
    symbol: str
    asset_class: str


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int
    backoff_seconds: tuple[int, ...]


@dataclass(frozen=True)
class IngestConfig:
    primary_source: str
    retry: RetryPolicy
    stale_threshold_days: int
    outlier_intraday_pct: float
    outlier_intraday_pct_by_class: dict[str, float] = field(default_factory=dict)
    # Design §1.7: signals are computed only if at least this share of the
    # NYSE-aligned universe fetched cleanly; below it the run is degraded.
    min_coverage_pct: float = 90.0

    def outlier_threshold(self, asset_class: str) -> float:
        return self.outlier_intraday_pct_by_class.get(asset_class, self.outlier_intraday_pct)


@dataclass(frozen=True)
class StorageConfig:
    duckdb_path: Path


@dataclass(frozen=True)
class ScheduleConfig:
    daily_at: str               # "HH:MM"
    timezone: str               # IANA TZ name; passed to zoneinfo.ZoneInfo
    weekdays_only: bool
    run_on_startup: bool
    catch_up_missed: bool


@dataclass(frozen=True)
class ReportsConfig:
    """Reports auto-commit (O1) — off by default; user opts in via config.yaml.

    - auto_commit: if true, after both US+HK PDFs ship to Telegram, git-commit
      just that day's report artifacts (no -A, no push).
    - auto_commit_skip_on_degraded: skip the commit when the run logged
      `degraded` (signals not published) — degraded reports shouldn't pollute
      the history."""
    auto_commit: bool = False
    auto_commit_skip_on_degraded: bool = True


@dataclass(frozen=True)
class Config:
    storage: StorageConfig
    ingest: IngestConfig
    schedule: ScheduleConfig | None = None
    universe: tuple[Symbol, ...] = field(default_factory=tuple)
    reports: ReportsConfig = field(default_factory=ReportsConfig)

    def symbols(self) -> list[str]:
        return [s.symbol for s in self.universe]

    def asset_class(self, symbol: str) -> str:
        for s in self.universe:
            if s.symbol == symbol:
                return s.asset_class
        raise KeyError(symbol)


_UNIVERSE_KEYS = (
    "equities",
    "defensives",
    "growth_sensitive",
    "crypto",
    "signal_support",
    "china_us_listed",
    "hk_listed",
)


def load_config(path: str | Path = "config.yaml") -> Config:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))

    universe: list[Symbol] = []
    for key in _UNIVERSE_KEYS:
        for entry in raw.get("universe", {}).get(key, []):
            universe.append(Symbol(symbol=entry["symbol"], asset_class=entry["asset_class"]))

    ingest = raw["ingest"]

    sched = None
    if "schedule" in raw and raw["schedule"]:
        s = raw["schedule"]
        sched = ScheduleConfig(
            daily_at=str(s.get("daily_at", "22:00")),
            timezone=str(s.get("timezone", "UTC")),
            weekdays_only=bool(s.get("weekdays_only", True)),
            run_on_startup=bool(s.get("run_on_startup", False)),
            catch_up_missed=bool(s.get("catch_up_missed", False)),
        )

    reports_raw = raw.get("reports", {}) or {}
    reports = ReportsConfig(
        auto_commit=bool(reports_raw.get("auto_commit", False)),
        auto_commit_skip_on_degraded=bool(
            reports_raw.get("auto_commit_skip_on_degraded", True)
        ),
    )

    return Config(
        storage=StorageConfig(duckdb_path=Path(raw["storage"]["duckdb_path"])),
        ingest=IngestConfig(
            primary_source=ingest["primary_source"],
            retry=RetryPolicy(
                max_attempts=ingest["retry"]["max_attempts"],
                backoff_seconds=tuple(ingest["retry"]["backoff_seconds"]),
            ),
            stale_threshold_days=ingest["stale_threshold_days"],
            outlier_intraday_pct=ingest["outlier_intraday_pct"],
            outlier_intraday_pct_by_class=dict(ingest.get("outlier_intraday_pct_by_class", {})),
            min_coverage_pct=float(ingest.get("min_coverage_pct", 90.0)),
        ),
        schedule=sched,
        universe=tuple(universe),
        reports=reports,
    )
