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

import os
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
