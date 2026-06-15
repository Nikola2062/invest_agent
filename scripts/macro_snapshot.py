"""Market-wide macro / crash-risk snapshot for the TradingAgents report.

Fetches eight macro indicators, classifies each with a traffic light, tallies
them into an overall risk level, and reports the S&P 500 drawdown plus a
cash-deployment action band. This is *market context* (not per-ticker): it is
computed once per batch and rendered at the top of the combined HTML report.

The indicator set and thresholds are ported from the standalone
``investor_agent/risk_analysis`` tool so the two stay consistent.

Design: every fetcher is independently wrapped and returns ``None`` on any
failure (network down, layout change, missing key). A degraded snapshot still
renders — missing indicators simply show "N/A" and are excluded from the tally.
Two indicators (HY OAS, Sahm Rule) need a free FRED API key; without it they
show "N/A (needs fred_api_key)" and the report is otherwise unaffected.
"""
from __future__ import annotations

import csv
import io
import re
import time
from datetime import datetime

import requests
import yfinance as yf
from bs4 import BeautifulSoup

UA = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36"
    )
}
TIMEOUT = 30


def _bounded(value, lo, hi, name):
    """Return ``value`` if within [lo, hi]; otherwise log a warning and return None.

    Guards against a source layout change making a regex/parse grab a
    wrong-but-authoritative number (worse than N/A): an out-of-range value is
    dropped to None so it shows "N/A" and is excluded from the tally.
    """
    if value is None:
        return None
    if value < lo or value > hi:
        print(f"  ! {name} value {value} outside plausible range [{lo}, {hi}] — treating as N/A")
        return None
    return value


# ------------------------------- Fetchers ------------------------------------

def fetch_shiller_cape():
    try:
        r = requests.get("https://www.multpl.com/shiller-pe", headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        current = soup.find(id="current")
        if current:
            m = re.search(r"([\d.]+)", current.get_text())
            if m:
                return _bounded(float(m.group(1)), 5, 70, "Shiller CAPE"), None
    except Exception as e:
        print(f"  ! Shiller CAPE fetch failed: {e}")
    return None, None


def fetch_buffett_indicator():
    try:
        r = requests.get(
            "https://www.currentmarketvaluation.com/models/buffett-indicator.php",
            headers=UA, timeout=TIMEOUT,
        )
        r.raise_for_status()
        text = r.text
        m = re.search(r"Buffett\s+Indicator\s+as\s+(\d{2,3})\s*%", text, re.I)
        if not m:
            m = re.search(r"Buffett\s+Indicator[^%]{0,40}?(\d{2,3})\s*%", text, re.I)
        if m:
            return _bounded(float(m.group(1)), 50, 400, "Buffett Indicator"), None
    except Exception as e:
        print(f"  ! Buffett Indicator fetch failed: {e}")
    return None, None


def fetch_treasury_t10y2y():
    """10Y-2Y spread from the US Treasury daily yield-curve CSV."""
    try:
        year = datetime.now().year
        url = (
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
            f"daily-treasury-rates.csv/{year}/all"
            f"?type=daily_treasury_yield_curve&field_tdr_date_value={year}&_format=csv"
        )
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        rows = list(csv.DictReader(io.StringIO(r.text)))
        if not rows:
            return None, None
        latest = rows[0]
        spread = float(latest["10 Yr"]) - float(latest["2 Yr"])
        return _bounded(spread, -3, 5, "10Y-2Y spread"), latest.get("Date")
    except Exception as e:
        print(f"  ! T10Y2Y fetch failed: {e}")
    return None, None


def fetch_fred_api(series_id, api_key):
    """Latest value of a FRED series via the official JSON API (needs a key)."""
    if not api_key:
        return None, None
    url = (
        "https://api.stlouisfed.org/fred/series/observations"
        f"?series_id={series_id}&api_key={api_key}"
        "&file_type=json&sort_order=desc&limit=20"
    )
    last_err = None
    for delay in (0, 2, 4):
        if delay:
            time.sleep(delay)
        try:
            r = requests.get(url, headers=UA, timeout=60)
            r.raise_for_status()
            for obs in r.json().get("observations", []):
                if obs.get("value") and obs["value"] != ".":
                    return float(obs["value"]), obs.get("date")
            return None, None
        except Exception as e:
            last_err = e
    print(f"  ! FRED API {series_id} fetch failed after 3 attempts: {last_err}")
    return None, None


def fetch_cnn_fear_greed():
    try:
        r = requests.get(
            "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            headers=UA, timeout=TIMEOUT,
        )
        r.raise_for_status()
        fg = r.json().get("fear_and_greed", {})
        return _bounded(float(fg.get("score")), 0, 100, "CNN Fear & Greed"), fg.get("rating")
    except Exception as e:
        print(f"  ! CNN F&G fetch failed: {e}")
    return None, None


def fetch_margin_debt_yoy(manual=None):
    """YoY change in FINRA margin debit balances (latest vs same month a year prior)."""
    if manual is not None:
        return float(manual)
    try:
        r = requests.get(
            "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics",
            headers=UA, timeout=TIMEOUT,
        )
        r.raise_for_status()
        table = BeautifulSoup(r.text, "html.parser").find("table")
        if not table:
            return None
        data = {}
        for tr in table.find_all("tr")[1:]:
            cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
            if len(cells) < 2:
                continue
            try:
                data[cells[0]] = float(cells[1].replace(",", ""))
            except ValueError:
                continue
        if not data:
            return None
        latest_key = list(data.keys())[0]  # e.g. "Apr-26"
        mon, yy = latest_key.split("-")
        prior_key = f"{mon}-{int(yy) - 1:02d}"
        if prior_key not in data:
            return None
        yoy = (data[latest_key] - data[prior_key]) / data[prior_key] * 100.0
        return _bounded(yoy, -100, 200, "Margin Debt YoY")
    except Exception as e:
        print(f"  ! Margin debt YoY fetch failed: {e}")
    return None


def fetch_ism_pmi(manual=None, below_50_months=0):
    if manual is not None:
        return float(manual), int(below_50_months)
    try:
        r = requests.get(
            "https://tradingeconomics.com/united-states/business-confidence",
            headers=UA, timeout=TIMEOUT,
        )
        r.raise_for_status()
        m = re.search(r"PMI[^\d]{0,40}(\d{2}\.\d)", r.text)
        if m:
            return _bounded(float(m.group(1)), 30, 70, "ISM PMI"), int(below_50_months)
    except Exception as e:
        print(f"  ! ISM PMI fetch failed: {e}")
    return None, int(below_50_months)


def fetch_vix():
    """Latest close of the Cboe VIX (^VIX) via yfinance."""
    try:
        hist = yf.Ticker("^VIX").history(period="5d", auto_adjust=False)
        if hist.empty:
            return None, None
        return _bounded(float(hist["Close"].iloc[-1]), 5, 150, "Cboe VIX"), hist.index[-1].strftime("%Y-%m-%d")
    except Exception as e:
        print(f"  ! VIX fetch failed: {e}")
    return None, None


def fetch_sp500_drawdown():
    """Return (current_close, all_time_high, drawdown_pct) for the S&P 500."""
    try:
        hist = yf.Ticker("^GSPC").history(period="max", auto_adjust=False)
        if hist.empty:
            return None, None, None
        current = float(hist["Close"].iloc[-1])
        ath = float(hist["Close"].max())
        return current, ath, (current - ath) / ath * 100.0
    except Exception as e:
        print(f"  ! S&P 500 fetch failed: {e}")
    return None, None, None


# ------------------------------ Classifiers ----------------------------------

def _light(color, label):
    return {"emoji": color, "label": label}

def classify_cape(v):
    if v is None: return _light("⚪", "N/A")
    if v > 30: return _light("🔴", "Red")
    if v >= 20: return _light("🟡", "Yellow")
    return _light("🟢", "Green")

def classify_buffett(v):
    if v is None: return _light("⚪", "N/A")
    if v > 150: return _light("🔴", "Red")
    if v >= 100: return _light("🟡", "Yellow")
    return _light("🟢", "Green")

def classify_t10y2y(v):
    if v is None: return _light("⚪", "N/A")
    if v < 0: return _light("🔴", "Red")
    if v <= 0.50: return _light("🟡", "Yellow")
    return _light("🟢", "Green")

def classify_margin_debt(v):
    if v is None: return _light("⚪", "N/A")
    if v > 30: return _light("🔴", "Red")
    if v >= 0: return _light("🟡", "Yellow")
    return _light("🟢", "Green")

def classify_fear_greed(v):
    if v is None: return _light("⚪", "N/A")
    if v > 90: return _light("🔴", "Deep Red")
    if v > 75: return _light("🔴", "Red")
    return _light("🟢", "Green")

def classify_hy_oas(v):
    if v is None: return _light("⚪", "N/A")
    if v > 6: return _light("🔴", "Red")
    if v >= 4: return _light("🟡", "Yellow")
    return _light("🟢", "Green")

def classify_vix(v):
    """U-shaped: <15 complacency (Yellow) / 15-25 Green / 25-30 Yellow / >30 Red."""
    if v is None: return _light("⚪", "N/A")
    if v > 30: return _light("🔴", "Red")
    if v >= 25: return _light("🟡", "Yellow")
    if v < 15: return _light("🟡", "Yellow")
    return _light("🟢", "Green")

def classify_macro(pmi, pmi_below_50_months, sahm):
    if pmi is None and sahm is None:
        return _light("⚪", "N/A")
    if pmi is not None and pmi < 50 and pmi_below_50_months >= 3:
        return _light("🔴", "Red")
    if sahm is not None and sahm > 0.5:
        return _light("🔴", "Red")
    return _light("🟢", "Green")


# ------------------------------- Action band ---------------------------------

def action_for_drawdown(dd_pct):
    if dd_pct is None:
        return "N/A — data unavailable"
    if dd_pct > -10:
        return "HOLD — do not deploy cash, do not chase."
    if dd_pct > -20:
        return "PREPARE CASH — raise cash buffer; pre-write your buy list."
    if dd_pct > -30:
        return "DEPLOY 1/3 of cash into target allocations."
    if dd_pct > -40:
        return "DEPLOY another 1/3 of cash."
    return "DEPLOY ALL remaining cash."


# ------------------------------- Snapshot ------------------------------------

def build_snapshot(fred_key: str | None = None, manual_overrides: dict | None = None) -> dict:
    """Fetch all indicators and return the macro snapshot dict.

    Resilient by construction: any individual fetch failure yields an "N/A"
    indicator rather than raising, so the snapshot always builds.
    """
    overrides = manual_overrides or {}
    fred_key = fred_key or ""

    cape, _ = fetch_shiller_cape()
    buffett, _ = fetch_buffett_indicator()
    t10y2y, t10y2y_date = fetch_treasury_t10y2y()
    margin_yoy = fetch_margin_debt_yoy(overrides.get("margin_debt_yoy_pct"))
    fg_score, fg_rating = fetch_cnn_fear_greed()
    vix_level, vix_date = fetch_vix()
    hy_oas, hy_date = fetch_fred_api("BAMLH0A0HYM2", fred_key)
    hy_oas = _bounded(hy_oas, 0, 25, "HY OAS")
    sahm, sahm_date = fetch_fred_api("SAHMREALTIME", fred_key)
    sahm = _bounded(sahm, -1, 5, "Sahm Rule")
    pmi, pmi_below_50_months = fetch_ism_pmi(
        overrides.get("ism_pmi"),
        overrides.get("ism_pmi_below_50_months", 0),
    )
    sp_current, sp_ath, sp_dd = fetch_sp500_drawdown()

    indicators = [
        {"name": "Shiller CAPE", "value": cape,
         "display": f"{cape:.2f}" if cape is not None else "N/A",
         "threshold": ">30 Red / 20-30 Yellow", "light": classify_cape(cape)},
        {"name": "Buffett Indicator", "value": buffett,
         "display": f"{buffett:.0f}%" if buffett is not None else "N/A",
         "threshold": ">150% Red / 100-150% Yellow", "light": classify_buffett(buffett)},
        {"name": "10Y-2Y Spread", "value": t10y2y,
         "display": f"{t10y2y:+.2f}% ({t10y2y_date})" if t10y2y is not None else "N/A",
         "threshold": "<0 Red / 0-0.50% Yellow", "light": classify_t10y2y(t10y2y)},
        {"name": "Margin Debt YoY", "value": margin_yoy,
         "display": f"{margin_yoy:+.2f}%" if margin_yoy is not None else "N/A",
         "threshold": ">30% Red / 0-30% Yellow", "light": classify_margin_debt(margin_yoy)},
        {"name": "CNN Fear & Greed", "value": fg_score,
         "display": f"{fg_score:.0f} ({fg_rating})" if fg_score is not None else "N/A",
         "threshold": ">75 Red / >90 Deep Red", "light": classify_fear_greed(fg_score)},
        {"name": "Cboe VIX", "value": vix_level,
         "display": f"{vix_level:.2f} ({vix_date})" if vix_level is not None else "N/A",
         "threshold": ">30 Red / 25-30 Yellow / 15-25 Green / <15 Yellow",
         "light": classify_vix(vix_level)},
        {"name": "HY OAS", "value": hy_oas,
         "display": f"{hy_oas:.2f}% ({hy_date})" if hy_oas is not None else "N/A (needs fred_api_key)",
         "threshold": ">6% Red / 4-6% Yellow", "light": classify_hy_oas(hy_oas)},
        {"name": "PMI + Sahm Rule",
         "value": {"pmi": pmi, "sahm": sahm, "pmi_below_50_months": pmi_below_50_months},
         "display": (f"PMI={pmi if pmi is not None else 'N/A'}, "
                     f"Sahm={sahm if sahm is not None else 'N/A'}"),
         "threshold": "PMI<50 (3mo) OR Sahm>0.5",
         "light": classify_macro(pmi, pmi_below_50_months, sahm)},
    ]

    # Stamp manually-overridden indicators so a stale hand-entered value is visible.
    # Manual values are trusted (not bounded) but must be dated.
    as_of = overrides.get("as_of")
    stamp = " · manual" + (f" (as of {as_of})" if as_of else "")
    _manual_names = {
        "margin_debt_yoy_pct": "Margin Debt YoY",
        "ism_pmi": "PMI + Sahm Rule",
    }
    for key, ind_name in _manual_names.items():
        if overrides.get(key) is not None:
            for ind in indicators:
                if ind["name"] == ind_name:
                    ind["display"] += stamp

    reds = sum(1 for ind in indicators if ind["light"]["emoji"] == "🔴")
    yellows = sum(1 for ind in indicators if ind["light"]["emoji"] == "🟡")
    greens = sum(1 for ind in indicators if ind["light"]["emoji"] == "🟢")

    if reds <= 1:
        risk = "LOW"
    elif reds <= 3:
        risk = "MEDIUM"
    elif reds <= 5:
        risk = "HIGH"
    else:
        risk = "CRITICAL"

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "date": datetime.now().strftime("%Y-%m-%d"),
        "indicators": indicators,
        "tally": {"red": reds, "yellow": yellows, "green": greens},
        "risk_level": risk,
        "sp500": {"current": sp_current, "ath": sp_ath, "drawdown_pct": sp_dd},
        "action": action_for_drawdown(sp_dd),
    }


if __name__ == "__main__":  # pragma: no cover - manual smoke test
    import json
    import os
    snap = build_snapshot(fred_key=os.environ.get("FRED_API_KEY"))
    print(json.dumps(snap, indent=2, ensure_ascii=False))
