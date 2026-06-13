"""Adapter for capital_rotation_analysier.

Read-only access to rotation.duckdb. The US regime + 8 composite signals + US
per-symbol returns live in the DB. HK is excluded from the NYSE-aligned signal
panel, so HK movers + a lightweight HK regime proxy are computed here directly
from raw_bars (mirrors report_hk's heuristic: benchmark 21d return sign +
breadth of HK names positive over 5d).
"""
from __future__ import annotations

import math
import os
import subprocess
from pathlib import Path

import duckdb

import settings

_CFG = settings.CONFIG


def _db_path() -> Path:
    return settings.project_dir("rotation") / _CFG["projects"]["rotation"]["duckdb"]


def available() -> bool:
    return _db_path().exists()


def run_daily(timeout: int = 1800) -> tuple[bool, str]:
    """Run capital_rotation's own daily pipeline to refresh the DB.

    Inherits credentials from the environment (DEEPSEEK_API_KEY, FRED_API_KEY)
    but STRIPS the Telegram creds so rotation does not send its own alerts —
    the orchestrator owns the single unified digest.
    """
    proj = settings.project_dir("rotation")
    py = settings.project_python("rotation")
    if not py.exists():
        return False, f"no venv python at {py}"
    env = dict(os.environ)
    env["PYTHONPATH"] = "src"
    env.pop("TELEGRAM_BOT_TOKEN", None)
    env.pop("TELEGRAM_CHAT_ID", None)
    cmd = [str(py)] + list(_CFG["projects"]["rotation"]["daily_cmd"])
    try:
        r = subprocess.run(cmd, cwd=str(proj), capture_output=True, text=True,
                           timeout=timeout, env=env)
        return r.returncode == 0, (r.stdout + r.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        return False, "timeout"


def _log_return(series: list[float], lag: int) -> float | None:
    """Log return over `lag` rows; series is oldest->newest closes."""
    if len(series) <= lag or series[-1 - lag] <= 0 or series[-1] <= 0:
        return None
    return math.log(series[-1] / series[-1 - lag])


def fetch(hk_symbols: list[str] | None = None) -> dict:
    """Return the latest rotation snapshot.

    {
      asof, us_regime{label,confidence,days_in_regime},
      signals{name:{score,confidence}},
      us_movers{symbol:{r_d,r_w,r_m}},
      hk{asof, movers{symbol:{r_d,r_w,r_m}}, regime_proxy{label,benchmark_r_m,breadth}}
    }
    """
    if not available():
        return {"available": False, "reason": f"missing {_db_path()}"}

    con = duckdb.connect(str(_db_path()), read_only=True)
    try:
        out: dict = {"available": True}

        reg = con.execute(
            "SELECT ts, regime, confidence, days_in_regime "
            "FROM regime_history ORDER BY ts DESC LIMIT 1"
        ).fetchone()
        if reg:
            out["asof"] = str(reg[0])
            out["us_regime"] = {
                "label": reg[1],
                "confidence": reg[2],
                "days_in_regime": reg[3],
            }

        sig_rows = con.execute(
            "SELECT signal_name, score, confidence FROM signals_daily "
            "WHERE ts = (SELECT MAX(ts) FROM signals_daily)"
        ).fetchall()
        out["signals"] = {r[0]: {"score": r[1], "confidence": r[2]} for r in sig_rows}

        # US per-symbol returns (exclude HK names, which aren't in the NYSE panel).
        mrows = con.execute(
            "SELECT symbol, r_d, r_w, r_m FROM metrics_daily "
            "WHERE ts = (SELECT MAX(ts) FROM metrics_daily)"
        ).fetchall()
        out["us_movers"] = {
            r[0]: {"r_d": r[1], "r_w": r[2], "r_m": r[3]} for r in mrows
        }

        # --- HK: computed directly from raw_bars ---------------------------
        hk_symbols = hk_symbols or []
        hk: dict = {"movers": {}}
        hk_max = con.execute(
            "SELECT MAX(ts) FROM raw_bars WHERE asset_class = 'equity_hk'"
        ).fetchone()[0]
        if hk_max is not None:
            hk["asof"] = str(hk_max)
            # All HK symbols present, for breadth.
            all_hk = [
                r[0]
                for r in con.execute(
                    "SELECT DISTINCT symbol FROM raw_bars WHERE asset_class='equity_hk'"
                ).fetchall()
            ]
            pos_5d = 0
            counted = 0
            for sym in all_hk:
                closes = [
                    r[0]
                    for r in con.execute(
                        "SELECT adj_close FROM raw_bars WHERE symbol=? AND asset_class='equity_hk' "
                        "ORDER BY ts ASC",
                        [sym],
                    ).fetchall()
                    if r[0] is not None
                ]
                rw = _log_return(closes, 5)
                if rw is not None:
                    counted += 1
                    if rw > 0:
                        pos_5d += 1
                if sym in hk_symbols:
                    hk["movers"][sym] = {
                        "r_d": _log_return(closes, 1),
                        "r_w": rw,
                        "r_m": _log_return(closes, 21),
                    }
            # Regime proxy from the HSI tracker (2800.HK) + breadth.
            bench = [
                r[0]
                for r in con.execute(
                    "SELECT adj_close FROM raw_bars WHERE symbol='2800.HK' AND asset_class='equity_hk' "
                    "ORDER BY ts ASC"
                ).fetchall()
                if r[0] is not None
            ]
            bench_rm = _log_return(bench, 21)
            breadth = (pos_5d / counted) if counted else None
            label = "Uncertain"
            if bench_rm is not None and breadth is not None:
                if bench_rm > 0 and breadth >= 0.55:
                    label = "Risk-On"
                elif bench_rm < 0 and breadth <= 0.45:
                    label = "Risk-Off"
                else:
                    label = "Mixed"
            hk["regime_proxy"] = {
                "label": label,
                "benchmark_r_m": bench_rm,
                "breadth": breadth,
            }
        out["hk"] = hk
        return out
    finally:
        con.close()
