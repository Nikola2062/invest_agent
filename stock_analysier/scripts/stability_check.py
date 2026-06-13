"""Run the pipeline N times on a single ticker, report cross-run variance.

The deterministic tactical_exit ladder consumes LLM-produced
risk.drawdown_probabilities. If those four numbers drift across same-day reruns,
the discipline layer is sitting on shaky ground. This script measures that drift.

Defaults: 5 runs on 0700.HK, no audit persistence (so the experiment doesn't
pollute the production DB).

Usage:
    PYTHONPATH=. .venv/bin/python scripts/stability_check.py
    PYTHONPATH=. .venv/bin/python scripts/stability_check.py --symbol MSFT --market US --runs 3
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline.orchestrator import analyze, reset_macro_cache  # noqa: E402

log = logging.getLogger("stability_check")


def _extract_metrics(result) -> dict:
    """Pull the LLM-driven scalars worth tracking across runs."""
    p_dd = result.risk.drawdown_probabilities
    fs = result.forward_scenarios
    da = result.devil_advocate
    return {
        "p_dd_10": float(p_dd.get("10", 0.0)),
        "p_dd_15": float(p_dd.get("15", 0.0)),
        "p_dd_20": float(p_dd.get("20", 0.0)),
        "p_dd_25": float(p_dd.get("25", 0.0)),
        "realized_vol_pct": result.risk.realized_vol_annualized_pct,
        "tactical_level": result.if_held.tactical.level,
        "tactical_label": result.if_held.tactical.label,
        "if_not_held_rec": result.if_not_held.recommendation,
        "intrinsic_low": result.valuation.intrinsic_low,
        "intrinsic_base": result.valuation.intrinsic_base,
        "intrinsic_high": result.valuation.intrinsic_high,
        "margin_of_safety_pct": result.valuation.margin_of_safety_pct,
        "fwd_expected_return_pct": fs.expected_return_pct if fs else None,
        "fwd_prob_weighted_target": fs.probability_weighted_target if fs else None,
        "da_verdict": da.overall_verdict if da else None,
        "da_finding_count": len(da.findings) if da else None,
        "quality_score": result.fundamental.quality_score,
        "moat_strength": result.fundamental.moat_strength,
    }


def _summarize_numeric(values: list, label: str) -> str:
    cleaned = [v for v in values if v is not None and isinstance(v, (int, float)) and not math.isnan(v)]
    if not cleaned:
        return f"  {label:<28} (no numeric data)"
    mn, mx = min(cleaned), max(cleaned)
    mean = statistics.fmean(cleaned)
    stdev = statistics.pstdev(cleaned) if len(cleaned) > 1 else 0.0
    span = mx - mn
    rel = (span / abs(mean) * 100) if mean else 0.0
    return (
        f"  {label:<28} mean={mean:>9.4f}  stdev={stdev:>8.4f}  "
        f"min={mn:>9.4f}  max={mx:>9.4f}  span={span:>8.4f}  ({rel:>5.1f}% of |mean|)"
    )


def _summarize_categorical(values: list, label: str) -> str:
    counts: dict = {}
    for v in values:
        key = repr(v)
        counts[key] = counts.get(key, 0) + 1
    parts = [f"{k}×{c}" for k, c in sorted(counts.items(), key=lambda kv: -kv[1])]
    return f"  {label:<28} {', '.join(parts)}"


def main(symbol: str, market: str, runs: int, persist: bool) -> int:
    logging.basicConfig(
        level=logging.WARNING,  # quiet — we print our own progress
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
    )
    print(f"\n=== Stability check: {symbol} ({market}), {runs} runs, persist={persist} ===\n")
    snapshots = []
    overall_t0 = time.time()

    for i in range(runs):
        t0 = time.time()
        print(f"[run {i+1}/{runs}] starting…", flush=True)
        try:
            reset_macro_cache()       # don't share the LRU across runs
            result = analyze(symbol, market, persist=persist)
            dur = time.time() - t0
            metrics = _extract_metrics(result)
            snapshots.append({"run": i + 1, "duration_s": dur, **metrics})
            print(
                f"[run {i+1}/{runs}] done in {dur:.1f}s  "
                f"P(dd≥15%)={metrics['p_dd_15']:.3f}  "
                f"tactical={metrics['tactical_label']!s}  "
                f"DA={metrics['da_verdict']!s}",
                flush=True,
            )
        except Exception as e:
            log.exception("run %d failed", i + 1)
            snapshots.append({"run": i + 1, "error": str(e)})
            print(f"[run {i+1}/{runs}] FAILED: {e}", flush=True)

    total = time.time() - overall_t0

    # ---- write raw snapshots ----
    out_path = PROJECT_ROOT / "data" / f"stability_check_{symbol.replace('.', '_')}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "symbol": symbol, "market": market, "runs": runs,
        "started_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_duration_s": total,
        "snapshots": snapshots,
    }, indent=2, default=str))
    print(f"\nRaw snapshots written to: {out_path}")

    # ---- summary ----
    successful = [s for s in snapshots if "error" not in s]
    print(f"\n=== Summary ({len(successful)}/{runs} successful, total {total:.1f}s) ===\n")
    if not successful:
        print("No successful runs to summarize.")
        return 1

    numeric_keys = [
        "p_dd_10", "p_dd_15", "p_dd_20", "p_dd_25",
        "intrinsic_low", "intrinsic_base", "intrinsic_high",
        "margin_of_safety_pct", "fwd_expected_return_pct",
        "fwd_prob_weighted_target", "quality_score", "da_finding_count",
    ]
    cat_keys = ["tactical_level", "tactical_label", "if_not_held_rec",
                "da_verdict", "moat_strength"]

    print("Numeric metrics (variance across runs):")
    for k in numeric_keys:
        vals = [s.get(k) for s in successful]
        print(_summarize_numeric(vals, k))

    print("\nCategorical metrics (distribution across runs):")
    for k in cat_keys:
        vals = [s.get(k) for s in successful]
        print(_summarize_categorical(vals, k))

    # ---- interpretation cue ----
    p15 = [s["p_dd_15"] for s in successful]
    if len(p15) > 1:
        span = max(p15) - min(p15)
        print(f"\nP(dd≥15%) span across runs: {span:.3f}")
        if span >= 0.20:
            print("→ ⚠️  LARGE variance. The tactical ladder is sitting on noisy LLM probabilities.")
            print("   Different reruns same day will push you into different YELLOW/ORANGE/RED states.")
        elif span >= 0.10:
            print("→ ⚠️  Moderate variance. Borderline cases will flip-flop between reruns.")
        else:
            print("→ ✓  Low variance. The deterministic ladder has a stable input.")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--symbol", default="0700.HK")
    parser.add_argument("--market", default="HK", choices=["US", "HK"])
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--persist", action="store_true",
                        help="Persist runs to the audit DB (default: don't pollute)")
    args = parser.parse_args()
    sys.exit(main(args.symbol, args.market, args.runs, args.persist))
