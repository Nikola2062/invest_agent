from __future__ import annotations

import duckdb

RAW_BARS_DDL = """
CREATE TABLE IF NOT EXISTS raw_bars (
    symbol        VARCHAR  NOT NULL,
    asset_class   VARCHAR  NOT NULL,
    ts            DATE     NOT NULL,
    open          DOUBLE,
    high          DOUBLE,
    low           DOUBLE,
    close         DOUBLE,
    adj_close     DOUBLE,
    volume        DOUBLE,
    source        VARCHAR  NOT NULL,
    revision      INTEGER  NOT NULL DEFAULT 0,
    ingested_at   TIMESTAMP NOT NULL,
    stale         BOOLEAN  NOT NULL DEFAULT FALSE,
    PRIMARY KEY (symbol, ts)
);
"""

RAW_BARS_QUARANTINE_DDL = """
CREATE TABLE IF NOT EXISTS raw_bars_quarantine (
    symbol        VARCHAR,
    asset_class   VARCHAR,
    ts            DATE,
    open          DOUBLE, high DOUBLE, low DOUBLE, close DOUBLE,
    adj_close     DOUBLE, volume DOUBLE,
    source        VARCHAR,
    reason        VARCHAR,
    quarantined_at TIMESTAMP
);
"""

ETF_FLOWS_DDL = """
CREATE TABLE IF NOT EXISTS etf_flows (
    symbol              VARCHAR  NOT NULL,
    ts                  DATE     NOT NULL,
    shares_outstanding  DOUBLE,
    aum_usd             DOUBLE,
    net_flow_usd        DOUBLE,
    source              VARCHAR  NOT NULL,
    proxy_method        VARCHAR  NOT NULL,
    confidence          DOUBLE   NOT NULL,
    ingested_at         TIMESTAMP NOT NULL,
    PRIMARY KEY (symbol, ts)
);
"""

SIGNAL_VALIDATION_DDL = """
CREATE TABLE IF NOT EXISTS signal_validation (
    signal_name        VARCHAR  NOT NULL,
    asof_date          DATE     NOT NULL,
    verdict            VARCHAR  NOT NULL,   -- 'pass' | 'fail' | 'undetermined'
    reason             VARCHAR,             -- failure mode if not pass
    median_ic_5d       DOUBLE,
    median_ic_21d      DOUBLE,
    pct_windows_pos_ic DOUBLE,              -- fraction of rolling windows with IC > 0
    hit_rate_overall   DOUBLE,
    hit_rate_low_vol   DOUBLE,
    hit_rate_mid_vol   DOUBLE,
    hit_rate_high_vol  DOUBLE,
    n_observations     INTEGER,
    forward_asset      VARCHAR,             -- which asset's forward return we tested against
    details            JSON,
    computed_at        TIMESTAMP NOT NULL,
    PRIMARY KEY (signal_name, asof_date)
);
"""

SIGNALS_DAILY_DDL = """
CREATE TABLE IF NOT EXISTS signals_daily (
    ts          DATE     NOT NULL,
    signal_name VARCHAR  NOT NULL,
    score       DOUBLE,
    confidence  DOUBLE,
    components  JSON,
    computed_at TIMESTAMP NOT NULL,
    PRIMARY KEY (ts, signal_name)
);
"""

METRICS_DAILY_DDL = """
CREATE TABLE IF NOT EXISTS metrics_daily (
    ts          DATE     NOT NULL,
    symbol      VARCHAR  NOT NULL,
    r_d         DOUBLE,
    r_w         DOUBLE,
    r_m         DOUBLE,
    r_q         DOUBLE,
    vol_30      DOUBLE,
    vol_ratio   DOUBLE,
    rv          DOUBLE,
    vz          DOUBLE,
    rs_rank     INTEGER,
    rs_change_1 INTEGER,
    rs_change_5 INTEGER,
    rs_change_21 INTEGER,
    rs_accel_5  INTEGER,
    computed_at TIMESTAMP NOT NULL,
    PRIMARY KEY (ts, symbol)
);
"""

# Idempotent column additions for existing databases that pre-date the
# rs_change_1 / rs_accel_5 columns introduced in Phase A. DuckDB supports
# IF NOT EXISTS on ALTER TABLE ADD COLUMN.
METRICS_DAILY_MIGRATIONS = [
    "ALTER TABLE metrics_daily ADD COLUMN IF NOT EXISTS rs_change_1 INTEGER",
    "ALTER TABLE metrics_daily ADD COLUMN IF NOT EXISTS rs_accel_5  INTEGER",
]

# W1 Forecast Scorecard. One row per published forecast; the resolution
# columns are filled once the horizon has elapsed. The scorecard section
# reports hit rates over the resolved rows.
FORECASTS_DDL = """
CREATE TABLE IF NOT EXISTS forecasts (
    ts             DATE     NOT NULL,   -- publish date (signal asof)
    forecast_type  VARCHAR  NOT NULL,   -- 'spy_5d' | 'spy_21d' | 'sector_21d'
    horizon_days   INTEGER  NOT NULL,   -- trading days
    target         VARCHAR,             -- ticker (spy_*) or NULL (sector)
    direction      VARCHAR,             -- bullish | neutral | bearish (dominant bucket)
    bullish_pct    DOUBLE,
    neutral_pct    DOUBLE,
    bearish_pct    DOUBLE,
    median_fwd     DOUBLE,              -- analogue median forward log return
    cutoff         DOUBLE,              -- |log return| above which a window is directional
    confidence     DOUBLE,              -- forecast_confidence score 0..1
    n_analogues    INTEGER,
    details        JSON,                -- e.g. {"predicted_top3": [...]} for sector
    created_at     TIMESTAMP NOT NULL,
    resolved_at    TIMESTAMP,           -- NULL until the horizon elapses
    actual_value   DOUBLE,              -- realized log return (spy_*) or NULL
    actual_direction VARCHAR,           -- bullish | neutral | bearish
    hit            BOOLEAN,
    PRIMARY KEY (ts, forecast_type)
);
"""

REGIME_HISTORY_DDL = """
CREATE TABLE IF NOT EXISTS regime_history (
    ts          DATE     NOT NULL PRIMARY KEY,
    regime      VARCHAR  NOT NULL,
    prev_regime VARCHAR,
    confidence  DOUBLE,
    days_in_regime INTEGER,
    components  JSON,
    computed_at TIMESTAMP NOT NULL
);
"""

REPORTS_DDL = """
CREATE TABLE IF NOT EXISTS reports (
    ts          DATE     NOT NULL,
    horizon     VARCHAR  NOT NULL,
    body_md     TEXT     NOT NULL,
    generated_at TIMESTAMP NOT NULL,
    PRIMARY KEY (ts, horizon)
);
"""

ALERTS_DDL = """
CREATE TABLE IF NOT EXISTS alerts (
    alert_id    VARCHAR  NOT NULL PRIMARY KEY,
    ts          DATE     NOT NULL,
    alert_type  VARCHAR  NOT NULL,
    priority    VARCHAR  NOT NULL,
    headline    VARCHAR  NOT NULL,
    body        TEXT,
    fired_at    TIMESTAMP NOT NULL,
    delivered   BOOLEAN  NOT NULL DEFAULT FALSE,
    channel     VARCHAR
);
"""

RUN_LOG_DDL = """
CREATE TABLE IF NOT EXISTS run_log (
    run_id      VARCHAR  NOT NULL,
    asof_date   DATE     NOT NULL,
    started_at  TIMESTAMP NOT NULL,
    finished_at TIMESTAMP,
    status      VARCHAR  NOT NULL,
    n_symbols   INTEGER,
    n_inserted  INTEGER,
    n_updated   INTEGER,
    n_quarantined INTEGER,
    notes       VARCHAR
);
"""


def ensure_schema(con: duckdb.DuckDBPyConnection) -> None:
    con.execute(RAW_BARS_DDL)
    con.execute(RAW_BARS_QUARANTINE_DDL)
    con.execute(ETF_FLOWS_DDL)
    con.execute(RUN_LOG_DDL)
    con.execute(METRICS_DAILY_DDL)
    con.execute(SIGNALS_DAILY_DDL)
    con.execute(SIGNAL_VALIDATION_DDL)
    con.execute(FORECASTS_DDL)
    con.execute(REGIME_HISTORY_DDL)
    con.execute(REPORTS_DDL)
    con.execute(ALERTS_DDL)
    for stmt in METRICS_DAILY_MIGRATIONS:
        con.execute(stmt)
