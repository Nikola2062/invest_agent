"""Runs INSIDE risk_analysis's venv (cwd = risk_analysis project root).

Calls build_report() directly so we get a fresh macro report WITHOUT going
through main.py's Telegram gate (which refuses to start headless when
telegram.enabled is true). Writes the report JSON via the project's own
save_report() and prints the path.

Usage:  python _macro_runner.py
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))  # cwd is the risk_analysis project root


def main() -> int:
    import main as ra_main  # risk_analysis/main.py
    from analyzer import build_report

    cfg = ra_main.load_config()          # loads .env, injects FRED_API_KEY
    report = build_report(cfg)
    path = ra_main.save_report(report, cfg.get("reports_dir", "reports"))
    print("<<<ORCH_JSON>>>")
    print(json.dumps({"path": str(path), "risk_level": report.get("risk_level")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
