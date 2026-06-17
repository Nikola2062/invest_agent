"""Load the user's book and apply the position-aware overlay to a batch of runs.

Bridges three things that are otherwise blind to each other:
  - the book (positions.yaml: holdings + watchlist + cash),
  - per-name ratings from the TradingAgents pipeline (or any decision text),
  - the market context (macro level + regime),
and produces, per name, a HOLDER action via ``position_overlay.evaluate``.

I/O lives here (file + yfinance prices); the decision logic stays pure in
``position_overlay``.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml

from tradingagents.agents.utils.rating import parse_rating
from tradingagents.portfolio import position_overlay as po

# Fallback HKD->USD if the live FX fetch fails (only affects concentration weights).
_HKD_USD_FALLBACK = 0.128


def load_positions(path: str | Path = "positions.yaml") -> dict:
    """Read the book. Returns {held: [...], watchlist: {US:[],HK:[]}, cash: [...]}.
    Missing file -> empty book (the overlay simply treats every name as not-held)."""
    p = Path(path)
    if not p.exists():
        return {"held": [], "watchlist": {}, "cash": []}
    data = yaml.safe_load(p.read_text()) or {}
    return {
        "held": data.get("holdings") or [],
        "watchlist": data.get("watchlist") or {},
        "cash": data.get("cash") or [],
    }


def _fx_to_usd(currency: str) -> float:
    if currency == "USD":
        return 1.0
    try:
        import yfinance as yf
        hist = yf.Ticker(f"{currency}USD=X").history(period="5d", auto_adjust=False)
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    return _HKD_USD_FALLBACK if currency == "HKD" else 1.0


def latest_prices(symbols: list[str]) -> dict[str, float]:
    """Latest close per symbol via yfinance. Best-effort: missing -> absent key."""
    out: dict[str, float] = {}
    if not symbols:
        return out
    try:
        import yfinance as yf
        for sym in symbols:
            try:
                hist = yf.Ticker(sym).history(period="5d", auto_adjust=False)
                if not hist.empty:
                    out[sym] = float(hist["Close"].iloc[-1])
            except Exception:
                continue
    except Exception:
        pass
    return out


def book_weights(held: list[dict], prices: dict[str, float]) -> dict[str, float]:
    """Each holding's % of total book market value, USD-normalised. Empty if no prices."""
    values: dict[str, float] = {}
    for h in held:
        sym = h.get("symbol")
        px = prices.get(sym)
        if px is None or not h.get("shares"):
            continue
        values[sym] = px * float(h["shares"]) * _fx_to_usd(h.get("currency", "USD"))
    total = sum(values.values())
    if total <= 0:
        return {}
    return {sym: v / total * 100.0 for sym, v in values.items()}


def overlay_for_runs(report_runs: list[dict], positions: dict, *,
                     macro_level: Optional[str] = None, regime: Optional[str] = None,
                     prices: Optional[dict[str, float]] = None,
                     cfg: Optional[po.OverlayConfig] = None) -> list[dict]:
    """Map each run's decision -> rating -> holder action.

    ``report_runs`` entries are the dicts main.py builds: {ticker, decision, ...}.
    Held names use cost basis + concentration weight; others get an entry reco.
    """
    held = {h["symbol"]: h for h in positions.get("held", [])}
    if prices is None:
        prices = latest_prices(list(held.keys()) + [r["ticker"] for r in report_runs])
    weights = book_weights(positions.get("held", []), prices)

    verdicts = []
    for run in report_runs:
        sym = run["ticker"]
        rating = parse_rating(str(run.get("decision", "")))
        h = held.get(sym)
        pos = po.Position(
            symbol=sym,
            rating=rating,
            held=h is not None,
            shares=float(h["shares"]) if h and h.get("shares") else 0.0,
            cost_basis=(float(h["cost_basis_per_share"]) if h and h.get("cost_basis_per_share") else None),
            current_price=prices.get(sym),
            weight_pct=weights.get(sym),
            purchase_date=(h.get("purchase_date") if h else None),
            market=(h.get("market") if h else ("HK" if sym.endswith(".HK") else "US")),
        )
        verdicts.append(po.evaluate(pos, macro_level=macro_level, regime=regime, cfg=cfg))
    return verdicts


# --------------------------- on-demand triage (cheap) ------------------------
# The scheduled overview is built with NO LLM ratings. ``triage_alarms`` decides,
# from the quant layer alone (drawdown-from-high, relative strength, market
# danger), which HELD names deserve an automatic full-pipeline deep-dive.
# Everything else is left for the user to pull on demand via the bot. Pure: every
# input is a value (drawdowns/ranking already fetched), so the whole function is
# unit-testable offline — same shape as ``position_overlay``.


@dataclass
class TriageConfig:
    """Thresholds for the deterministic deep-dive alarms."""
    drawdown_alarm_pct: float = -8.0   # this far below the recent HIGH -> look
    rs_alarm_score: float = -5.0       # relative strength this weak -> look
    risk_off_regimes: tuple = po.OverlayConfig().risk_off_regimes
    danger_macro_levels: tuple = po.OverlayConfig().danger_macro_levels


def drawdown_from_peak(closes) -> Optional[float]:
    """Latest close as a % below the running peak of a price series (<= 0.0).

    0.0 means "at a fresh high"; -12.3 means "12.3% off the recent high". Pure;
    accepts any iterable of numbers (NaNs dropped). None if no usable data.

    Drawdown-from-high is a *fresh-move* signal: unlike P&L vs cost basis it
    resets as the price recovers, so a chronically underwater holding does not
    trip the alarm on every run — only a genuine recent selloff does.
    """
    vals = [float(x) for x in closes if x == x]  # x == x drops NaN
    if not vals:
        return None
    peak = max(vals)
    if peak <= 0:
        return None
    return (vals[-1] / peak - 1.0) * 100.0


def recent_drawdowns(symbols: list[str], period: str = "3mo") -> dict[str, float]:
    """Drawdown-from-recent-high (%) per symbol via yfinance. Best-effort:
    missing/short/failed -> absent key (the alarm then can't trip for it)."""
    out: dict[str, float] = {}
    if not symbols:
        return out
    try:
        import yfinance as yf
    except Exception:
        return out
    for sym in symbols:
        try:
            hist = yf.Ticker(sym).history(period=period, auto_adjust=False)
            if hist.empty:
                continue
            dd = drawdown_from_peak(hist["Close"].tolist())
            if dd is not None:
                out[sym] = dd
        except Exception:
            continue
    return out


def triage_alarms(book: dict, drawdowns: dict[str, float],
                  ranking: Optional[list[dict]] = None,
                  regime: Optional[str] = None,
                  macro_level: Optional[str] = None,
                  cfg: Optional[TriageConfig] = None) -> list[dict]:
    """Which HELD names deserve an automatic deep-dive, and why.

    Returns ``[{symbol, reasons: [str, ...]}, ...]`` for held names that trip at
    least one alarm. Watchlist names are never auto-run (pull-only). A name with
    no usable drawdown or RS simply can't trip those rungs (fails to silence,
    never to a false alarm). Pure; no I/O — ``drawdowns`` is precomputed (see
    ``recent_drawdowns``) so the decision stays unit-testable offline.
    """
    cfg = cfg or TriageConfig()
    rs_by = {r["symbol"]: r for r in (ranking or [])}
    risk_off = (
        (regime in set(cfg.risk_off_regimes))
        or (bool(macro_level) and macro_level.upper()
            in {x.upper() for x in cfg.danger_macro_levels})
    )

    out: list[dict] = []
    for h in book.get("held", []):
        sym = h.get("symbol")
        if not sym:
            continue
        reasons: list[str] = []
        categories: list[str] = []

        dd = drawdowns.get(sym)
        if dd is not None and dd <= cfg.drawdown_alarm_pct:
            reasons.append(f"{dd:+.0f}% from high")
            categories.append("drawdown")

        rs = rs_by.get(sym)
        if rs is not None and rs.get("rs_score") is not None \
                and rs["rs_score"] <= cfg.rs_alarm_score:
            reasons.append(f"RS {rs['rs_score']:+.1f}")
            categories.append("rs")

        if risk_off:
            reasons.append("risk-off")
            categories.append("risk_off")

        if reasons:
            # ``categories`` + ``drawdown`` are machine-readable fields for the
            # dedupe (``select_for_deepdive``); the digest only reads symbol/reasons.
            out.append({"symbol": sym, "reasons": reasons,
                        "categories": categories, "drawdown": dd})
    return out


# ------------------------- per-name deep-dive dedupe -------------------------
# An alarm is a *standing state*, not an event: a name 25% off its high stays
# that way for days. Without dedupe it would auto-deep-dive on every fire. The
# dedupe re-runs a name only when its alarm is genuinely NEW or materially WORSE
# (or after a refresh interval), so steady-state alarms go quiet between fires.
# The decision is pure; only load/save state touch disk.


@dataclass
class DedupeConfig:
    """When an already-seen alarm warrants a fresh deep-dive again."""
    rerun_worsen_pct: float = 5.0   # drawdown must worsen >= this many pts to re-fire
    rerun_after_days: float = 5.0   # ...or this long since the last run (periodic refresh)


def _parse_iso(ts) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(ts) if ts else None
    except (ValueError, TypeError):
        return None


def select_for_deepdive(alarms: list[dict], state: dict, now: datetime,
                        cfg: Optional[DedupeConfig] = None) -> tuple[list[dict], dict]:
    """Filter alarmed names to those that warrant a FRESH deep-dive given prior
    run history. Pure.

    A name (re)runs if it is newly alarming, a new alarm category appeared, its
    drawdown worsened by >= ``rerun_worsen_pct``, or ``rerun_after_days`` have
    elapsed since its last run. Returns ``(to_run, new_state)``. ``new_state``
    holds ONLY currently-alarming names — a recovered name is dropped, so it
    counts as new if it alarms again later.
    """
    cfg = cfg or DedupeConfig()
    state = state or {}
    to_run: list[dict] = []
    new_state: dict = {}
    for a in alarms:
        sym = a["symbol"]
        prev = state.get(sym)
        cats = set(a.get("categories", []))
        dd = a.get("drawdown")
        rerun = False
        if prev is None:
            rerun = True                                          # never seen
        elif cats - set(prev.get("categories", [])):
            rerun = True                                          # new alarm category
        elif (dd is not None and prev.get("drawdown") is not None
              and dd <= prev["drawdown"] - cfg.rerun_worsen_pct):
            rerun = True                                          # drawdown worsened
        else:
            last = _parse_iso(prev.get("ts"))
            stale = True
            if last is not None:
                try:
                    stale = (now - last).total_seconds() >= cfg.rerun_after_days * 86400
                except TypeError:                                 # tz-naive vs aware
                    stale = True
            if stale:
                rerun = True                                      # periodic refresh

        if rerun:
            to_run.append(a)
            new_state[sym] = {"ts": now.isoformat(), "drawdown": dd,
                              "categories": sorted(cats)}
        else:
            new_state[sym] = prev                                 # keep last run's record
    return to_run, new_state


def load_triage_state(path: str | Path) -> dict:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        data = json.loads(p.read_text())
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save_triage_state(path: str | Path, state: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(state, indent=2))
