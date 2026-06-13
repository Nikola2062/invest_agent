"""F1 — Diagnose the Phase B finding that risk_on_off and inflation are
anti-correlated with their target asset at the 5d horizon over 10 years.

Per the project docs §1.6.2 the 5d-IC verdicts on the 10-year backfill were:
  - risk_on_off : median 5d IC = -0.076
  - inflation   : median 5d IC = -0.091

This script tests THREE candidate explanations:

  (A) Wrong horizon — the signal might be predictive at 21d or 63d but
      anti-predictive at 5d (lead/lag mismatch with the basket components).
  (B) Wrong target — the basket might predict a different asset than the
      one the validator defaults to. For risk_on_off → SPY is the assumed
      target; maybe it predicts QQQ or IWM or HYG better.
  (C) Wrong sign — the score might be set up with inverted polarity (i.e.
      the formula gives +ve for risk-off conditions).

Run from project root:
  .venv/bin/python scripts/investigate_basket_inversion.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

# Path setup for scripts/ invocation
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rotation.config import load_config                                # noqa: E402
from rotation.store import connect                                     # noqa: E402


def _load_signal(con, name: str) -> pd.Series:
    df = con.execute(
        "SELECT ts, score FROM signals_daily WHERE signal_name = ? "
        "AND score IS NOT NULL ORDER BY ts", [name]
    ).df()
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")["score"]


def _load_close(con, sym: str) -> pd.Series:
    df = con.execute(
        "SELECT ts, adj_close FROM raw_bars WHERE symbol = ? "
        "AND adj_close IS NOT NULL ORDER BY ts", [sym]
    ).df()
    df["ts"] = pd.to_datetime(df["ts"])
    return df.set_index("ts")["adj_close"]


def _fwd_log_return(close: pd.Series, h: int) -> pd.Series:
    return np.log(close.shift(-h) / close)


def _rolling_ic(score: pd.Series, fwd: pd.Series, window: int = 252) -> pd.Series:
    s = pd.concat([score.rename("s"), fwd.rename("f")], axis=1).dropna()
    if len(s) < window:
        return pd.Series(dtype=float)
    sr = s["s"].rolling(window).rank()
    fr = s["f"].rolling(window).rank()
    return sr.rolling(window).corr(fr)


def _summarise(ic: pd.Series) -> dict:
    if ic.empty or ic.dropna().empty:
        return {"median": None, "pct_pos": None, "n": 0}
    valid = ic.dropna()
    return {
        "median":  float(valid.median()),
        "pct_pos": float((valid > 0).mean()),
        "n":       int(len(valid)),
    }


def investigate(cfg, signal_name: str, candidate_targets: list[str], horizons=(5, 21, 63)) -> None:
    print(f"\n## {signal_name}")
    print()
    print("| target | h | median IC | %pos | n |")
    print("|---|---:|---:|---:|---:|")
    with connect(cfg.storage.duckdb_path) as con:
        score = _load_signal(con, signal_name)
        if score.empty:
            print("_no signal history_")
            return
        for tgt in candidate_targets:
            close = _load_close(con, tgt)
            if close.empty:
                print(f"| {tgt} | — | _no bars_ |  |  |")
                continue
            for h in horizons:
                fwd = _fwd_log_return(close, h)
                ic = _rolling_ic(score, fwd, window=252)
                s = _summarise(ic)
                if s["n"] == 0:
                    print(f"| {tgt} | {h} | — | — | 0 |")
                else:
                    print(f"| {tgt} | {h} | {s['median']:+.3f} | "
                          f"{s['pct_pos']*100:.0f}% | {s['n']} |")


def main():
    cfg = load_config(Path(__file__).resolve().parent.parent / "config.yaml")

    print("# Basket Inversion Investigation")
    print()
    print("Per Phase B the IC harness reported negative median 5d IC for "
          "risk_on_off and inflation. This script tests whether the signal is "
          "predictive at OTHER horizons (5/21/63d) and against OTHER targets "
          "than the validator's default. A positive median IC on any cell "
          "means the signal predicts that asset over that horizon — a clear "
          "directional read.")

    investigate(
        cfg, "risk_on_off",
        candidate_targets=["SPY", "QQQ", "IWM", "HYG", "TLT", "GLD"],
    )

    investigate(
        cfg, "inflation",
        candidate_targets=["CPER", "USO", "GLD", "SLV", "TLT", "SPY"],
    )

    investigate(
        cfg, "capital_rotation",
        candidate_targets=["SPY", "QQQ", "IWM", "TLT"],
    )

    investigate(
        cfg, "growth",
        candidate_targets=["SMH", "XLI", "IYT", "SPY", "QQQ", "XLU"],
    )

    investigate(
        cfg, "liquidity",
        candidate_targets=["SPY", "QQQ", "HYG", "BTC-USD", "TLT"],
    )

    print()
    print("---")
    print()
    print("**Reading guide:**")
    print()
    print("- Median IC ≥ +0.05 and %pos ≥ 60% on at least one (target, horizon) "
          "cell means the signal is genuinely predictive there — the 5d-vs-SPY "
          "test was just the wrong test.")
    print("- Negative median IC across ALL targets and horizons means the signal "
          "polarity is genuinely inverted — the basket construction needs review.")
    print("- Roughly zero median IC across targets but a positive hit rate at one "
          "horizon means the signal is magnitude-informative but not direction-"
          "informative; the analogue layer can still consume it via cosine sim.")


if __name__ == "__main__":
    main()
