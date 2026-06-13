"""Fetches the 8 crash-risk indicators, classifies them, and builds the report."""
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


# ---------- Fetchers ----------

def fetch_shiller_cape():
    try:
        r = requests.get("https://www.multpl.com/shiller-pe", headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        current = soup.find(id="current")
        if current:
            m = re.search(r"([\d.]+)", current.get_text())
            if m:
                return float(m.group(1)), None
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
        # Pattern: "we calculate the Buffett Indicator as 219%"
        m = re.search(r"Buffett\s+Indicator\s+as\s+(\d{2,3})\s*%", text, re.I)
        if not m:
            m = re.search(r"Buffett\s+Indicator[^%]{0,40}?(\d{2,3})\s*%", text, re.I)
        if m:
            return float(m.group(1)), None
    except Exception as e:
        print(f"  ! Buffett Indicator fetch failed: {e}")
    return None, None


def fetch_treasury_t10y2y():
    """10Y-2Y spread, computed from the US Treasury daily yield curve CSV."""
    try:
        year = datetime.now().year
        url = (
            "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
            f"daily-treasury-rates.csv/{year}/all"
            f"?type=daily_treasury_yield_curve&field_tdr_date_value={year}&_format=csv"
        )
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        reader = csv.DictReader(io.StringIO(r.text))
        rows = list(reader)
        if not rows:
            return None, None
        latest = rows[0]
        y10 = float(latest["10 Yr"])
        y2 = float(latest["2 Yr"])
        return y10 - y2, latest.get("Date")
    except Exception as e:
        print(f"  ! T10Y2Y fetch failed: {e}")
    return None, None


def fetch_fred_api(series_id, api_key):
    """Use the official FRED JSON API. Requires a free api_key.

    Retries up to 3 times with 2s / 4s backoff and a 60s per-attempt timeout
    to ride out transient network slowness (FRED is sometimes sluggish from
    EU/non-US egress).
    """
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
            data = r.json()
            for obs in data.get("observations", []):
                if obs.get("value") and obs["value"] != ".":
                    return float(obs["value"]), obs.get("date")
            return None, None
        except Exception as e:
            last_err = e
    print(f"  ! FRED API {series_id} fetch failed after 3 attempts: {last_err}")
    return None, None


def fetch_cnn_fear_greed():
    try:
        url = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
        r = requests.get(url, headers=UA, timeout=TIMEOUT)
        r.raise_for_status()
        data = r.json()
        fg = data.get("fear_and_greed", {})
        return float(fg.get("score")), fg.get("rating")
    except Exception as e:
        print(f"  ! CNN F&G fetch failed: {e}")
    return None, None


def fetch_margin_debt_yoy(manual=None):
    """Parse FINRA's official margin-statistics table; compute YoY from latest vs same month one year prior."""
    if manual is not None:
        return float(manual)
    try:
        r = requests.get(
            "https://www.finra.org/rules-guidance/key-topics/margin-accounts/margin-statistics",
            headers=UA, timeout=TIMEOUT,
        )
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        table = soup.find("table")
        if not table:
            return None
        rows = table.find_all("tr")
        # Build {Mon-YY: debit_balance}
        data = {}
        for tr in rows[1:]:
            cells = [c.get_text(strip=True) for c in tr.find_all(["td", "th"])]
            if len(cells) < 2:
                continue
            month, debit = cells[0], cells[1].replace(",", "")
            try:
                data[month] = float(debit)
            except ValueError:
                continue
        if not data:
            return None
        # Latest = first row's month label
        months = list(data.keys())
        latest_key = months[0]  # e.g. "Apr-26"
        # Same month, prior year
        mon, yy = latest_key.split("-")
        prior_yy = f"{int(yy) - 1:02d}"
        prior_key = f"{mon}-{prior_yy}"
        if prior_key not in data:
            return None
        return (data[latest_key] - data[prior_key]) / data[prior_key] * 100.0
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
            return float(m.group(1)), int(below_50_months)
    except Exception as e:
        print(f"  ! ISM PMI fetch failed: {e}")
    return None, int(below_50_months)


def fetch_vix():
    """Latest close of the Cboe VIX (^VIX) via yfinance."""
    try:
        vix = yf.Ticker("^VIX")
        hist = vix.history(period="5d", auto_adjust=False)
        if hist.empty:
            return None, None
        latest_close = float(hist["Close"].iloc[-1])
        latest_date = hist.index[-1].strftime("%Y-%m-%d")
        return latest_close, latest_date
    except Exception as e:
        print(f"  ! VIX fetch failed: {e}")
    return None, None


def fetch_sp500_drawdown():
    """Return (current_close, ath_close, drawdown_pct) using max lookback for ATH."""
    try:
        sp = yf.Ticker("^GSPC")
        hist = sp.history(period="max", auto_adjust=False)
        if hist.empty:
            return None, None, None
        current = float(hist["Close"].iloc[-1])
        ath = float(hist["Close"].max())
        dd_pct = (current - ath) / ath * 100.0
        return current, ath, dd_pct
    except Exception as e:
        print(f"  ! S&P 500 fetch failed: {e}")
    return None, None, None


# ---------- Classifiers ----------

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
    """VIX is U-shaped: very low = complacency, very high = panic.

    <15 Yellow (complacency) / 15-25 Green / 25-30 Yellow / >30 Red.
    """
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


# ---------- Action band ----------

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


# ---------- Report builder ----------

def build_report(config):
    overrides = config.get("manual_overrides", {})
    fred_key = config.get("fred_api_key", "") or ""

    cape, _ = fetch_shiller_cape()
    buffett, _ = fetch_buffett_indicator()
    t10y2y, t10y2y_date = fetch_treasury_t10y2y()
    margin_yoy = fetch_margin_debt_yoy(overrides.get("margin_debt_yoy_pct"))
    fg_score, fg_rating = fetch_cnn_fear_greed()
    vix_level, vix_date = fetch_vix()
    hy_oas, hy_date = fetch_fred_api("BAMLH0A0HYM2", fred_key)
    sahm, sahm_date = fetch_fred_api("SAHMREALTIME", fred_key)
    pmi, pmi_below_50_months = fetch_ism_pmi(
        overrides.get("ism_pmi"),
        overrides.get("ism_pmi_below_50_months", 0),
    )
    sp_current, sp_ath, sp_dd = fetch_sp500_drawdown()

    indicators = [
        {
            "name": "Shiller CAPE",
            "value": cape,
            "display": f"{cape:.2f}" if cape is not None else "N/A",
            "threshold": ">30 Red / 20-30 Yellow",
            "light": classify_cape(cape),
        },
        {
            "name": "Buffett Indicator",
            "value": buffett,
            "display": f"{buffett:.0f}%" if buffett is not None else "N/A",
            "threshold": ">150% Red / 100-150% Yellow",
            "light": classify_buffett(buffett),
        },
        {
            "name": "10Y-2Y Spread",
            "value": t10y2y,
            "display": f"{t10y2y:+.2f}% ({t10y2y_date})" if t10y2y is not None else "N/A",
            "threshold": "<0 Red / 0-0.50% Yellow",
            "light": classify_t10y2y(t10y2y),
        },
        {
            "name": "Margin Debt YoY",
            "value": margin_yoy,
            "display": f"{margin_yoy:+.2f}%" if margin_yoy is not None else "N/A",
            "threshold": ">30% Red / 0-30% Yellow",
            "light": classify_margin_debt(margin_yoy),
        },
        {
            "name": "CNN Fear & Greed",
            "value": fg_score,
            "display": f"{fg_score:.0f} ({fg_rating})" if fg_score is not None else "N/A",
            "threshold": ">75 Red / >90 Deep Red",
            "light": classify_fear_greed(fg_score),
        },
        {
            "name": "Cboe VIX",
            "value": vix_level,
            "display": f"{vix_level:.2f} ({vix_date})" if vix_level is not None else "N/A",
            "threshold": ">30 Red / 25-30 Yellow / 15-25 Green / <15 Yellow",
            "light": classify_vix(vix_level),
        },
        {
            "name": "HY OAS",
            "value": hy_oas,
            "display": f"{hy_oas:.2f}% ({hy_date})" if hy_oas is not None else "N/A (needs fred_api_key)",
            "threshold": ">6% Red / 4-6% Yellow",
            "light": classify_hy_oas(hy_oas),
        },
        {
            "name": "PMI + Sahm Rule",
            "value": {"pmi": pmi, "sahm": sahm, "pmi_below_50_months": pmi_below_50_months},
            "display": (
                f"PMI={pmi if pmi is not None else 'N/A'}, "
                f"Sahm={sahm if sahm is not None else 'N/A'}"
            ),
            "threshold": "PMI<50 (3mo) OR Sahm>0.5",
            "light": classify_macro(pmi, pmi_below_50_months, sahm),
        },
    ]

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
        "sp500": {
            "current": sp_current,
            "ath": sp_ath,
            "drawdown_pct": sp_dd,
        },
        "action": action_for_drawdown(sp_dd),
    }


# ---------- Telegram formatter ----------

def format_telegram(report):
    lines = [f"📊 *US Equity Crash-Risk Report — {report['date']}*", ""]
    lines.append("*Indicators:*")
    for ind in report["indicators"]:
        lines.append(
            f"{ind['light']['emoji']} *{ind['name']}*: `{ind['display']}`\n"
            f"    _thr: {ind['threshold']}_"
        )
    lines.append("")
    t = report["tally"]
    lines.append(f"*Tally:* 🔴 {t['red']}  |  🟡 {t['yellow']}  |  🟢 {t['green']}")
    lines.append(f"*Risk Level:* *{report['risk_level']}*")
    lines.append("")
    sp = report["sp500"]
    if sp["drawdown_pct"] is not None:
        lines.append(
            f"*S&P 500:* {sp['current']:.2f}  "
            f"(ATH {sp['ath']:.2f}, {sp['drawdown_pct']:+.2f}%)"
        )
    else:
        lines.append("*S&P 500:* N/A")
    lines.append(f"*Action:* {report['action']}")
    return "\n".join(lines)
