"""Glossary of metric short-forms and ticker symbols used in the daily report.

The glossary is rendered as the final report section. Metrics are always shown
(short list, always relevant). Tickers are filtered to only those that actually
appear in the rendered body, so the glossary stays focused per-report.
"""
from __future__ import annotations

import re


# ---------- Metric / signal short-forms ----------

METRIC_GLOSSARY: dict[str, str] = {
    # Returns
    "r_d":         "Daily log return — ln(close_t / close_{t-1}).",
    "r_w":         "Weekly log return — 5 trading days. Rolling, not calendar week.",
    "r_m":         "Monthly log return — 21 trading days.",
    "r_q":         "Quarterly log return — 63 trading days. Used for regime input.",

    # Volatility
    "vol_30":      "Realized volatility, 30-day window, annualized (× √252).",
    "vol_ratio":   "Vol-of-vol — σ(10d) / σ(60d). >1.5 flags a vol regime break.",

    # Volume
    "RV":          "Relative Volume — ln(today's volume / 30-day median). Log-transformed so spikes are well-behaved.",
    "RV (log)":    "Relative Volume in log space — same as RV; column header style.",
    "vz":          "Volume z-score on LOG volume (raw V is log-normal; z-scoring raw V produces fat-tailed noise).",
    "log-V z-score": "Same as vz.",

    # Cross-sectional rank
    "RS rank":     "Relative Strength rank, 1–99, IBD-style cross-sectional percentile of blended-horizon return (0.4·r_m + 0.3·r_w + 0.2·r_q + 0.1·r_d). At N=17 assets the rank is interpolated and best read as a coarse leader/laggard tag.",
    "ΔRS(1d)":     "Change in RS rank over the past 1 trading day. Velocity at the shortest horizon — noisy by design, useful for spotting same-day flips.",
    "ΔRS(5d)":     "Change in RS rank over the past 5 trading days. Velocity at the canonical horizon.",
    "ΔRS(21d)":    "Change in RS rank over the past 21 trading days.",
    "Δ²RS(5d)":    "Acceleration — change in 5d-velocity vs 5d ago. Positive Δ²RS = leadership building; negative = leadership rolling over. Often turns before raw RS does.",

    # Phase A bucket / flow concepts
    "Bucket":              "Editorial group of related symbols (Technology = XLK+SMH; Crypto = BTC+ETH; etc.). Used by the Capital Flow Dashboard and Flow Map.",
    "Rotation Strength":   "|capital_rotation| → 0-100, with percentile vs trailing 252d. Tells you how unusual today's rotation magnitude is.",
    "risk tag":            "Bucket classification: risk_on / defensive / neutral / stress. Used by Flow Map to color-code direction.",

    # Composite signal scores (the 8 from §3.2)
    "Relative Strength":  "Cross-sectional z of blended-horizon return; headline is the top |z| asset (per-asset values in the components).",
    "Relative Volume":    "Universe-level activity score (0–100). High = aggregate volume well above 30-day baseline.",
    "Capital Rotation":   "5-pair block momentum spread (equities-vs-bonds, growth-vs-defensives, semis-vs-staples, US-vs-international, crypto-vs-gold). Positive = rotation into risk.",
    "Risk On Off":        "Risk-on basket (SPY, QQQ, IWM, EEM, BTC-USD, HYG) minus risk-off basket (TLT, GLD, UUP, FXF). Mapped to ±100 via tanh. Positive = risk-on.",
    "Inflation":          "Weighted z of copper/oil/gold/silver/bonds returns + breakeven if available. Positive = inflation rising.",
    "Growth":             "Growth leaders (SMH, XLI, IYT, CPER, EEM) minus laggards (XLP, XLU, TLT). Positive = growth accelerating.",
    "Recession":          "Yield curve proxy + HYG/LQD credit spread + XLY/XLP + IYT trend + UUP trend, signed toward recession. Slow signal at daily frequency.",
    "Liquidity":          "0–100 magnitude composite of UUP-inv + SPY-vol-inv + BTC + MOVE-inv + RRP-inv (the last two are added when MOVE / FRED are wired).",

    # Confidence
    "confidence (high)":  "≥ 0.70 — multiple agreeing inputs across horizons.",
    "confidence (medium)":"0.50–0.69 — meaningful but mixed.",
    "confidence (low)":   "0.35–0.49 — narrative downgraded to 'tentatively consistent with'.",
    "confidence (below floor)": "< 0.35 — narrative suppressed per §4.5 anti-hallucination policy.",
}


# ---------- Ticker definitions (only those in our universe) ----------

TICKER_GLOSSARY: dict[str, str] = {
    # US equity broad / size factors
    "SPY":      "SPDR S&P 500 ETF — US large-cap equity (the headline US risk read).",
    "QQQ":      "Invesco QQQ Trust — Nasdaq-100 (tech-heavy growth proxy).",
    "IWM":      "iShares Russell 2000 ETF — US small-cap (size factor / domestic-cyclical tell).",

    # International equity
    "EZU":      "iShares MSCI Eurozone ETF — Eurozone large/mid-cap.",
    "EEM":      "iShares MSCI Emerging Markets ETF — broad EM equity.",
    "EWJ":      "iShares MSCI Japan ETF.",

    # Precious metals / commodities
    "GLD":      "SPDR Gold Shares — physical gold price proxy.",
    "SLV":      "iShares Silver Trust — physical silver price proxy.",
    "CPER":     "United States Copper Index Fund — copper price proxy. Listed name 'Dr. Copper' for its growth-sensitivity.",
    "USO":      "United States Oil Fund — WTI crude price proxy.",

    # Bonds / rates
    "TLT":      "iShares 20+ Year Treasury Bond ETF — long duration; safe-haven and curve-flattener proxy.",
    "IEF":      "iShares 7–10 Year Treasury Bond ETF — intermediate duration. Paired with TLT for curve reads.",

    # Credit
    "HYG":      "iShares iBoxx $ High Yield Corporate Bond ETF — credit risk-on tell.",
    "LQD":      "iShares iBoxx $ Investment Grade Corporate Bond ETF — IG credit; paired with HYG for credit-spread reads.",

    # FX
    "UUP":      "Invesco DB US Dollar Index Bullish Fund — DXY (USD index) proxy. DXY ticker itself isn't on yfinance.",
    "FXF":      "Invesco CurrencyShares Swiss Franc Trust — safe-haven currency.",

    # US sector SPDRs (the 11 GICS sectors via State Street)
    "XLK":      "Technology Select Sector SPDR Fund.",
    "XLF":      "Financial Select Sector SPDR Fund.",
    "XLE":      "Energy Select Sector SPDR Fund.",
    "XLV":      "Health Care Select Sector SPDR Fund.",
    "XLY":      "Consumer Discretionary Select Sector SPDR Fund — risk-on consumer.",
    "XLP":      "Consumer Staples Select Sector SPDR Fund — defensive consumer.",
    "XLU":      "Utilities Select Sector SPDR Fund — defensive / rate-sensitive.",
    "XLB":      "Materials Select Sector SPDR Fund.",
    "XLI":      "Industrial Select Sector SPDR Fund — cyclical / capex-sensitive.",
    "XLRE":     "Real Estate Select Sector SPDR Fund.",
    "XLC":      "Communication Services Select Sector SPDR Fund.",

    # Industry ETFs
    "SMH":      "VanEck Semiconductor ETF — semis as leading-edge growth proxy.",
    "IYT":      "iShares US Transportation ETF — Dow Theory leading indicator.",

    # Crypto
    "BTC-USD":  "Bitcoin / US Dollar — high-beta liquidity proxy.",
    "ETH-USD":  "Ethereum / US Dollar.",

    # Volatility index ticker (CBOE/ICE), included with tickers since it's queried as a symbol on yfinance.
    "^MOVE":    "ICE BofA MOVE Index — implied volatility of US Treasury options. The 'bond-market VIX'.",

    # US-listed China proxies (Phase E)
    "KWEB":     "KraneShares CSI China Internet ETF — US-listed proxy for China internet names (Tencent, Alibaba, Meituan via H-shares).",
    "FXI":      "iShares China Large-Cap ETF — top 50 mainland-listed Chinese companies via HK.",
    "MCHI":     "iShares MSCI China ETF — broader China large/mid-cap (~600 names).",
    "ASHR":     "Xtrackers Harvest CSI 300 China A-Shares ETF — onshore A-share access via Stock Connect.",

    # HK-listed (Phase E, separate calendar — see Section 13)
    "2800.HK":  "Tracker Fund of Hong Kong — Hang Seng Index ETF.",
    "2828.HK":  "Hang Seng China Enterprises ETF (HSCEI / H-shares index).",
    "0700.HK":  "Tencent Holdings — HK-listed Chinese tech / internet giant.",
    "9988.HK":  "Alibaba Group — HK-listed (primary listing since 2019).",
    "3690.HK":  "Meituan — HK-listed Chinese local-services / food-delivery.",
    "1810.HK":  "Xiaomi — HK-listed Chinese consumer electronics.",
    "9618.HK":  "JD.com — HK-listed Chinese e-commerce.",
    "9888.HK":  "Baidu — HK-listed Chinese search / AI.",
    "2318.HK":  "Ping An Insurance — HK-listed Chinese insurance / fintech.",
    "0939.HK":  "China Construction Bank — HK-listed, one of the 'Big Four' Chinese banks.",
    "1398.HK":  "Industrial and Commercial Bank of China (ICBC) — HK-listed, largest Chinese bank by assets.",
    "0388.HK":  "Hong Kong Exchanges and Clearing (HKEX) — operator of the HK stock exchange.",
    "0005.HK":  "HSBC Holdings — HK-listed global bank with deep Asia exposure.",
    "0883.HK":  "CNOOC — HK-listed Chinese national offshore oil company.",
    "0857.HK":  "PetroChina — HK-listed Chinese state oil giant.",
}


# Tokens that look like tickers but should NOT be picked up as ticker references.
# Curated to avoid false positives from words inside the report body (e.g., "AS",
# "OK", market-vocabulary terms, day-of-week abbreviations).
_TICKER_BLACKLIST = {
    "OK", "FAILED", "AS", "VS", "AND", "OR", "NO", "TO", "IS", "ON", "OF",
    "FOR", "IN", "BY", "AT", "PER", "USD", "EUR", "JPY", "Y", "M", "D",
    "Q", "MAX", "MIN", "AVG", "ETF", "NYSE", "GICS", "DXY", "VIX", "WTI",
    "GFM", "AUM", "NAV", "CPI", "ISM", "NFP", "RRP", "FRED", "API", "JSON",
    "CSV", "URL", "PDF", "MD", "HTML", "CSS", "TLS", "SSL",
}


# Alternation is leftmost-match in Python regex; put the longest pattern
# (TICKER-USD, HK tickers) first so we don't match "BTC" before reaching
# "BTC-USD" or "0700" before reaching "0700.HK".
_TICKER_REGEX = re.compile(
    r"(?:[A-Z]+-USD|\d{4}\.HK|\^[A-Z]{3,6}|\b[A-Z]{2,5}\b)"
)


def find_tickers_in(text: str) -> set[str]:
    """Return the set of tickers from TICKER_GLOSSARY that appear in `text`.

    Uses an upper-case word-boundary regex, then intersects with the known
    glossary keys so we don't accidentally include unrelated all-caps tokens.
    """
    seen: set[str] = set()
    for match in _TICKER_REGEX.findall(text):
        if match in _TICKER_BLACKLIST:
            continue
        if match in TICKER_GLOSSARY:
            seen.add(match)
    return seen


def find_metrics_in(text: str) -> list[str]:
    """Metrics are listed in display order: returns first, then vol, then volume,
    rank, signals, confidence buckets. Filter to those appearing in `text`."""
    present = []
    # Cheap substring scan — metric keys may contain spaces/special chars
    for term in METRIC_GLOSSARY:
        # Word-boundary-ish: don't match inside another token, but allow at line
        # starts and after table-separator '|'.
        if re.search(r"(^|[\s\|()])" + re.escape(term) + r"($|[\s\|().,:])", text):
            present.append(term)
    return present


def render_glossary(report_body: str, section_number: int = 14) -> str:
    """Build the glossary section using only the terms that appear in the
    rendered report body."""
    tickers_present = sorted(find_tickers_in(report_body))
    metrics_present = find_metrics_in(report_body)

    out = [f"\n## Section {section_number} — Glossary\n"]
    out.append("_Only abbreviations actually used in this report are listed; the full universe glossary lives in `glossary.py`._\n")

    if metrics_present:
        out.append("### Metrics & signal short-forms")
        out.append("| Term | Meaning |")
        out.append("|---|---|")
        for term in metrics_present:
            defn = METRIC_GLOSSARY[term].replace("|", "\\|")
            out.append(f"| `{term}` | {defn} |")
        out.append("")

    if tickers_present:
        out.append("### Tickers referenced")
        out.append("| Ticker | Description |")
        out.append("|---|---|")
        for sym in tickers_present:
            defn = TICKER_GLOSSARY[sym].replace("|", "\\|")
            out.append(f"| `{sym}` | {defn} |")

    return "\n".join(out)
