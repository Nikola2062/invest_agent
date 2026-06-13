"""Adapter for stock_analysier (single-name deep dive).

Two paths:
  * read_latest(symbol)  — read the most recent run from audit.sqlite (no API
    calls; works entirely offline against already-collected data).
  * analyze_fresh(symbol, market) — run the project's analyze() in its OWN venv
    (subprocess) for an on-demand report; needs DeepSeek + network. Returns the
    project's own Telegram-formatted text plus headline fields.
"""
from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

import yaml

import settings

_CFG = settings.CONFIG

_FIELDS = (
    "symbol", "market", "timestamp_utc", "current_price", "currency",
    "quality_score", "margin_of_safety_pct", "tactical_level", "tactical_label",
    "if_held_action", "if_not_held_recommendation", "devil_verdict",
    "composite_tech_signal",
)


def _audit_db() -> Path:
    return settings.project_dir("stocks") / _CFG["projects"]["stocks"]["audit_db"]


def available() -> bool:
    return _audit_db().exists()


def read_positions() -> dict:
    """The single source of truth for held positions + watchlist: read directly
    from stock_analysier's own config (config/portfolio.yaml + config/universe.yaml).

    Returns {"held": [{symbol, market}], "watchlist": [{symbol, market}]}.
    """
    cfg_dir = settings.project_dir("stocks") / "config"

    def _load(name):
        p = cfg_dir / name
        return (yaml.safe_load(p.read_text()) if p.exists() else {}) or {}

    pf = _load("portfolio.yaml")
    uni = _load("universe.yaml")
    held = [
        {"symbol": h["symbol"], "market": h.get("market", "US")}
        for h in (pf.get("holdings") or [])
        if isinstance(h, dict) and h.get("symbol")
    ]
    watch = []
    for market, items in (uni.get("watchlist") or {}).items():
        for it in (items or []):
            sym = it.get("symbol") if isinstance(it, dict) else it
            if sym:
                watch.append({"symbol": sym, "market": market})
    return {"held": held, "watchlist": watch}


def read_latest(symbol: str) -> dict | None:
    """Most recent analysis_runs row for a symbol, as a plain dict (no full JSON)."""
    if not available():
        return None
    con = sqlite3.connect(f"file:{_audit_db()}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    try:
        row = con.execute(
            f"SELECT {', '.join(_FIELDS)} FROM analysis_runs "
            "WHERE symbol = ? ORDER BY timestamp_utc DESC LIMIT 1",
            [symbol],
        ).fetchone()
        return dict(row) if row else None
    finally:
        con.close()


def read_latest_full(symbol: str) -> dict | None:
    """The most recent run's full AnalysisResult (parsed full_result_json)."""
    if not available():
        return None
    con = sqlite3.connect(f"file:{_audit_db()}?mode=ro", uri=True)
    try:
        row = con.execute(
            "SELECT full_result_json FROM analysis_runs WHERE symbol = ? "
            "ORDER BY timestamp_utc DESC LIMIT 1",
            [symbol],
        ).fetchone()
        return json.loads(row[0]) if row else None
    finally:
        con.close()


def read_book(book: list[dict]) -> list[dict]:
    """For each {symbol,market} in book, attach its latest audit row (or None).

    Returns list of {symbol, market, latest|None}.
    """
    out = []
    for item in book:
        latest = read_latest(item["symbol"])
        out.append({"symbol": item["symbol"], "market": item["market"], "latest": latest})
    return out


def analyze_fresh(symbol: str, market: str, persist: bool = True, timeout: int = 900) -> dict:
    """Run stock_analysier.analyze() for one symbol via its own venv.

    Returns {ok, symbol, market, current_price, tactical_label, recommendation,
    telegram_text, error?}.
    """
    proj = settings.project_dir("stocks")
    py = settings.project_python("stocks")
    runner = settings.ORCH_DIR / _CFG["projects"]["stocks"]["runner"]
    if not py.exists():
        return {"ok": False, "error": f"no venv python at {py}"}
    if not runner.exists():
        return {"ok": False, "error": f"missing runner {runner}"}

    cmd = [str(py), str(runner), symbol, market]
    if not persist:
        cmd.append("--no-persist")
    try:
        r = subprocess.run(
            cmd, cwd=str(proj), capture_output=True, text=True, timeout=timeout
        )
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": "timeout", "symbol": symbol, "market": market}

    marker = "<<<ORCH_JSON>>>"
    if marker in r.stdout:
        payload = r.stdout.split(marker, 1)[1].strip()
        try:
            data = json.loads(payload)
            data["ok"] = True
            return data
        except json.JSONDecodeError as e:
            return {"ok": False, "error": f"bad json: {e}", "raw": payload[:500]}
    return {
        "ok": False,
        "error": "no result marker in output",
        "stderr": r.stderr[-1500:],
        "stdout": r.stdout[-500:],
        "symbol": symbol,
        "market": market,
    }
