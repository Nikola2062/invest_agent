"""Asset-class buckets used by the Phase A capital-flow renderers.

A bucket is a human-readable group label that aggregates one or more symbols
from the universe. Buckets exist so the report can answer "where is money
leaving / entering" at the level a human asks the question — sectors and
themes — rather than the level the model computes — individual tickers.

The mapping is intentionally redundant with `Symbol.asset_class` in
`config.yaml`: asset_class is engineering ("equity_sector", "bond"), bucket
is editorial ("Technology", "Long Bonds"). One ticker is in exactly one
bucket. A symbol that isn't mapped here is silently grouped under "Other".

Two views are exposed:
  - bucket_for(symbol) -> str
  - group_metrics_by_bucket(metrics_df) -> DataFrame indexed by bucket
"""
from __future__ import annotations

import pandas as pd


# Bucket → list of symbols. Order is presentation order.
BUCKETS: dict[str, list[str]] = {
    # ---- US equity, sector cuts ----
    "Technology":       ["XLK", "SMH"],
    "Communication":    ["XLC"],
    "Discretionary":    ["XLY"],
    "Financials":       ["XLF"],
    "Industrials":      ["XLI", "IYT"],
    "Materials":        ["XLB"],
    "Energy":           ["XLE", "USO"],
    "Real Estate":      ["XLRE"],
    "Healthcare":       ["XLV"],
    "Staples":          ["XLP"],
    "Utilities":        ["XLU"],
    # ---- US equity, size / market cuts ----
    "US Large Cap":     ["SPY", "QQQ"],
    "US Small Cap":     ["IWM"],
    # ---- International equity ----
    "Intl Developed":   ["EZU", "EWJ"],
    "Emerging Markets": ["EEM"],
    # ---- China (US-listed proxies) ----
    "China Internet":   ["KWEB"],                # KraneShares CSI China Internet ETF
    "China Large Cap":  ["FXI", "MCHI"],         # FXI = top 50 mainland-listed HK; MCHI = MSCI China
    "China A-Shares":   ["ASHR"],                # CSI 300 (mainland A-shares via Stock Connect)
    # ---- HK-listed individual stocks ----
    # These trade on HKEX so they don't enter the NYSE-aligned signals panel.
    # Bucketed here purely for the Greater China Holdings report section.
    "HK Internet":      ["0700.HK", "9988.HK", "3690.HK", "1810.HK", "9618.HK", "9888.HK"],
    "HK Financials":    ["2318.HK", "0939.HK", "1398.HK", "0388.HK", "0005.HK"],
    "HK Energy":        ["0883.HK", "0857.HK"],
    "HK Broad":         ["2800.HK", "2828.HK"],
    # ---- Crypto ----
    "Crypto":           ["BTC-USD", "ETH-USD"],
    # ---- Fixed income ----
    "Long Bonds":       ["TLT"],
    "Mid Bonds":        ["IEF"],
    "IG Credit":        ["LQD"],
    "HY Credit":        ["HYG"],
    # ---- Hard money / commodities ----
    "Gold":             ["GLD"],
    "Silver":           ["SLV"],
    "Copper":           ["CPER"],
    # ---- FX / safe-haven ----
    "USD":              ["UUP"],
    "Swiss Franc":      ["FXF"],
    # ---- Vol indices ----
    "Bond Volatility":  ["^MOVE"],
}


# Tag each bucket as risk-on / risk-off / neutral. Used by the flow-map
# renderer to color-code the direction of rotation.
BUCKET_RISK_TAG: dict[str, str] = {
    "Technology": "risk_on",
    "Communication": "risk_on",
    "Discretionary": "risk_on",
    "Financials": "risk_on",
    "Industrials": "risk_on",
    "Materials": "risk_on",
    "Energy": "neutral",          # behaves cyclically AND as inflation hedge
    "Real Estate": "neutral",
    "Healthcare": "defensive",
    "Staples": "defensive",
    "Utilities": "defensive",
    "US Large Cap": "risk_on",
    "US Small Cap": "risk_on",
    "Intl Developed": "risk_on",
    "Emerging Markets": "risk_on",
    "Crypto": "risk_on",
    "China Internet": "risk_on",
    "China Large Cap": "risk_on",
    "China A-Shares": "risk_on",
    "HK Internet": "risk_on",
    "HK Financials": "risk_on",
    "HK Energy": "neutral",
    "HK Broad": "risk_on",
    "Long Bonds": "defensive",
    "Mid Bonds": "defensive",
    "IG Credit": "defensive",
    "HY Credit": "risk_on",       # spread tightens on risk-on, widens on risk-off
    "Gold": "defensive",
    "Silver": "neutral",
    "Copper": "risk_on",
    "USD": "defensive",
    "Swiss Franc": "defensive",
    "Bond Volatility": "stress",  # rising MOVE = stress
}


# Reverse index, built once at import.
_SYMBOL_TO_BUCKET: dict[str, str] = {
    sym: bucket
    for bucket, syms in BUCKETS.items()
    for sym in syms
}


def bucket_for(symbol: str) -> str:
    """Bucket label for a symbol. Returns 'Other' for symbols not in the map."""
    return _SYMBOL_TO_BUCKET.get(symbol, "Other")


def risk_tag(bucket: str) -> str:
    return BUCKET_RISK_TAG.get(bucket, "neutral")


def group_metrics_by_bucket(metrics: pd.DataFrame) -> pd.DataFrame:
    """Aggregate a per-symbol metrics row into a per-bucket DataFrame.

    Input: metrics_daily row-set (one row per symbol) with at least
      `symbol`, `r_d`, `r_w`, `r_m`. Other numeric columns are averaged.

    Output: DataFrame indexed by bucket with columns:
      - `r_d_mean`, `r_w_mean`, `r_m_mean`: mean log return across constituents
      - `n`: number of constituents present
      - `pct_adv_w`: fraction of constituents with r_w > 0
      - `members`: comma-joined ticker list
      - `risk_tag`: risk_on / defensive / neutral / stress
    """
    if metrics.empty:
        return pd.DataFrame(
            columns=["r_d_mean", "r_w_mean", "r_m_mean", "n", "pct_adv_w",
                     "members", "risk_tag"]
        )
    m = metrics.copy()
    m["bucket"] = m["symbol"].map(bucket_for)

    g = m.groupby("bucket").agg(
        r_d_mean=("r_d", "mean"),
        r_w_mean=("r_w", "mean"),
        r_m_mean=("r_m", "mean"),
        n=("symbol", "size"),
        pct_adv_w=("r_w", lambda s: float((s > 0).sum()) / max(s.notna().sum(), 1)),
        members=("symbol", lambda s: ", ".join(sorted(s.tolist()))),
    )
    g["risk_tag"] = g.index.map(risk_tag)
    return g
