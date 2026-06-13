"""Adapter for risk_analysis (macro crash-risk gate).

Reads the newest report JSON the project writes to its reports/ dir. Can also
trigger a fresh one-shot run via the project's own venv (no Telegram). The
report schema (per risk_analysis): indicators[], tally, risk_level, sp500, action.
"""
from __future__ import annotations

import glob
import json
import subprocess
from pathlib import Path

import settings

_CFG = settings.CONFIG


def _reports_dir() -> Path:
    return settings.project_dir("macro_risk") / "reports"


def newest_report_path() -> Path | None:
    pattern = str(settings.project_dir("macro_risk") / _CFG["projects"]["macro_risk"]["reports_glob"])
    files = sorted(glob.glob(pattern))
    return Path(files[-1]) if files else None


def run_fresh(timeout: int = 600) -> tuple[bool, str]:
    """Run risk_analysis once (no Telegram) to produce a fresh report JSON.

    Drives build_report() via a runner so we bypass main.py's Telegram gate.
    Requires the project's venv to exist with its deps installed.
    """
    proj = settings.project_dir("macro_risk")
    py = settings.project_python("macro_risk")
    runner = settings.ORCH_DIR / _CFG["projects"]["macro_risk"]["runner"]
    if not py.exists():
        return False, f"no venv python at {py}"
    if not runner.exists():
        return False, f"missing runner {runner}"
    try:
        r = subprocess.run(
            [str(py), str(runner)],
            cwd=str(proj),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        ok = r.returncode == 0 and "<<<ORCH_JSON>>>" in r.stdout
        return ok, (r.stdout + r.stderr)[-2000:]
    except subprocess.TimeoutExpired:
        return False, "timeout"


def _indicator(report: dict, *needles: str) -> dict | None:
    """Find an indicator whose name contains any needle (case-insensitive)."""
    for ind in report.get("indicators", []):
        name = (ind.get("name") or "").lower()
        if any(n.lower() in name for n in needles):
            return ind
    return None


def fetch(fresh: bool = False) -> dict:
    """Return the macro gate snapshot. If fresh=True, run the project first."""
    if fresh:
        ok, log = run_fresh()
        if not ok:
            # fall through to newest existing report, but surface the failure
            pass

    path = newest_report_path()
    if path is None:
        return {"available": False, "reason": "no risk_analysis report on disk (run with --fresh once)"}

    report = json.loads(path.read_text())
    gauges = {}
    for key, needles in {
        "vix": ("vix",),
        "fear_greed": ("fear", "greed"),
        "cape": ("cape", "shiller"),
        "buffett": ("buffett",),
        "yield_curve": ("10y", "2y", "spread", "treasury"),
        "hy_oas": ("oas", "high yield", "hy "),
    }.items():
        ind = _indicator(report, *needles)
        if ind:
            gauges[key] = {
                "value": ind.get("value"),
                "display": ind.get("display"),
                "light": (ind.get("light") or {}).get("label"),
            }

    return {
        "available": True,
        "report_file": path.name,
        "date": report.get("date"),
        "risk_level": report.get("risk_level"),
        "tally": report.get("tally"),
        "action": report.get("action"),
        "sp500": report.get("sp500"),
        "gauges": gauges,
        "indicators": report.get("indicators", []),
    }
