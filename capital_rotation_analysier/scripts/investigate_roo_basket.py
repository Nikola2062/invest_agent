"""Per-constituent dissection of the risk_on_off basket.

Hypothesis under test:
  The score s_t = mean robust_z(r_w_ON) - mean robust_z(r_w_OFF) is built from
  *recent* (5d) returns. It then attempts to predict SPY's *next* 5d return.
  Because SPY itself sits in the ON basket and short-horizon equity returns
  are weakly mean-reverting at the 5d frequency, the headline score becomes
  structurally anti-correlated with its own target. We expect:

  (a) SPY and QQQ (high-beta, contained in ON) → individually NEGATIVE IC vs SPY fwd.
  (b) Off-basket members (TLT, GLD, UUP, FXF) → their inverted contribution should be
      POSITIVE IC if the diff structure works.
  (c) Removing SPY/QQQ/IWM from ON (i.e. predicting equities from
      non-equity risk-on assets) should restore positive IC.
  (d) Lengthening the formation horizon (5d → 21d → 63d) should also restore
      positive IC: the trend term dominates mean-reversion.

Outputs a markdown report of each test so we can pick the cleanest fix.

Run from project root:
  PYTHONPATH=src .venv/bin/python scripts/investigate_roo_basket.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rotation.config import load_config              # noqa: E402
from rotation.store import connect                   # noqa: E402
from rotation import metrics as M                    # noqa: E402


RISK_ON  = ["SPY", "QQQ", "IWM", "EEM", "BTC-USD", "HYG"]
RISK_OFF = ["TLT", "GLD", "UUP", "FXF"]


def _load_close_panel(con, symbols: list[str]) -> pd.DataFrame:
    df = con.execute(
        "SELECT symbol, ts, adj_close FROM raw_bars "
        "WHERE symbol IN ({}) AND adj_close IS NOT NULL "
        "AND asset_class != 'equity_hk' "
        "ORDER BY ts".format(",".join(f"'{s}'" for s in symbols))
    ).df()
    df["ts"] = pd.to_datetime(df["ts"])
    panel = df.pivot(index="ts", columns="symbol", values="adj_close").sort_index()
    # NYSE-align so crypto weekend bars (BTC-USD) don't poison the robust-z
    # rolling windows for the ETFs (same recipe as compute.load_panel).
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=panel.index.min(), end_date=panel.index.max())
    nyse_days = pd.DatetimeIndex(pd.to_datetime(sched.index).normalize())
    return panel.loc[panel.index.intersection(nyse_days)]


def _fwd_log_return(close: pd.Series, h: int) -> pd.Series:
    return np.log(close.shift(-h) / close)


def _ic_summary(score: pd.Series, fwd: pd.Series) -> dict:
    s = pd.concat([score.rename("s"), fwd.rename("f")], axis=1).dropna()
    if len(s) < 30:
        return {"spearman": None, "n": int(len(s))}
    rho = s["s"].rank().corr(s["f"].rank())
    return {"spearman": float(rho), "n": int(len(s))}


def _hit_rate(score: pd.Series, fwd: pd.Series) -> dict:
    """Fraction of days where sign(score) == sign(fwd)."""
    s = pd.concat([score.rename("s"), fwd.rename("f")], axis=1).dropna()
    if len(s) < 30:
        return {"hit": None, "n": int(len(s))}
    s = s[(s["s"].abs() > 0) & (s["f"].abs() > 0)]
    if len(s) == 0:
        return {"hit": None, "n": 0}
    hit = float((np.sign(s["s"]) == np.sign(s["f"])).mean())
    return {"hit": hit, "n": int(len(s))}


def _print_table(title: str, rows: list[tuple]) -> None:
    print(f"\n### {title}\n")
    print("| component | 5d IC | 5d hit | 21d IC | 21d hit | 63d IC | 63d hit | n |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        name, m5, h5, m21, h21, m63, h63, n = r
        fmt = lambda x: f"{x:+.3f}" if x is not None else "—"
        fmth = lambda x: f"{x*100:.0f}%" if x is not None else "—"
        print(f"| {name} | {fmt(m5)} | {fmth(h5)} | {fmt(m21)} | {fmth(h21)} | "
              f"{fmt(m63)} | {fmth(h63)} | {n} |")


def _ic_across_horizons(score: pd.Series, target: pd.Series) -> tuple:
    ic5 = _ic_summary(score, _fwd_log_return(target, 5))
    ic21 = _ic_summary(score, _fwd_log_return(target, 21))
    ic63 = _ic_summary(score, _fwd_log_return(target, 63))
    h5 = _hit_rate(score, _fwd_log_return(target, 5))
    h21 = _hit_rate(score, _fwd_log_return(target, 21))
    h63 = _hit_rate(score, _fwd_log_return(target, 63))
    return (ic5["spearman"], h5["hit"], ic21["spearman"], h21["hit"],
            ic63["spearman"], h63["hit"], ic5["n"])


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
    print("# risk_on_off basket dissection")
    print()
    print("Each row tests one component, signed in its natural ROO direction "
          "(ON members positive, OFF members negative — so that adding them up "
          "approximates the headline score). Score → SPY fwd log-return.")
    print()
    print(f"- ON basket : {RISK_ON}")
    print(f"- OFF basket: {RISK_OFF}")

    with connect(cfg.storage.duckdb_path) as con:
        close = _load_close_panel(con, RISK_ON + RISK_OFF)

    print(f"\n_Bars loaded: {close.shape[0]} sessions, "
          f"{close.shape[1]} symbols. Range: "
          f"{close.index.min().date()} → {close.index.max().date()}._")

    if "SPY" not in close.columns:
        print("\n_SPY not in panel — cannot compute target._")
        return

    target = close["SPY"]
    rw = M.log_returns(close, 5)         # the formation horizon ROO actually uses
    rm = M.log_returns(close, 21)        # alt formation horizon
    rq = M.log_returns(close, 63)        # alt formation horizon

    # 1. Per-constituent IC at the SAME formation horizon ROO uses (5d r_w).
    #    Sign each component as it enters the headline score: ON = +z, OFF = -z.
    rows = []
    for sym in RISK_ON:
        if sym not in close.columns:
            continue
        z = M.robust_z(rw[sym])
        rows.append((f"ON  {sym} (+z)",) + _ic_across_horizons(z, target))
    for sym in RISK_OFF:
        if sym not in close.columns:
            continue
        z = M.robust_z(rw[sym])
        rows.append((f"OFF {sym} (−z)",) + _ic_across_horizons(-z, target))
    _print_table("(1) Per-constituent IC vs SPY forward log-return — formation = r_w (5d)", rows)

    # 2. Headline ROO score reconstructed at 5d / 21d / 63d formation windows.
    def build_roo_score(returns: pd.DataFrame, on_syms: list[str], off_syms: list[str]) -> pd.Series:
        z = returns.apply(lambda c: M.robust_z(c), axis=0)
        on_z = z[[s for s in on_syms if s in z.columns]].mean(axis=1)
        off_z = z[[s for s in off_syms if s in z.columns]].mean(axis=1)
        diff = on_z - off_z
        return np.tanh(diff / 2.0) * 100.0

    rows = []
    for label, ret in (("5d", rw), ("21d", rm), ("63d", rq)):
        score = build_roo_score(ret, RISK_ON, RISK_OFF)
        rows.append((f"headline (formation={label})",) + _ic_across_horizons(score, target))
    _print_table("(2) Headline ROO at different formation horizons vs SPY fwd", rows)

    # 3. Drop one ON member at a time. If removing X turns the IC positive,
    #    X is poisoning the score. (Counter-test: drop one OFF member.)
    rows = []
    for drop in RISK_ON:
        on_minus = [s for s in RISK_ON if s != drop]
        score = build_roo_score(rw, on_minus, RISK_OFF)
        rows.append((f"ON drop {drop}",) + _ic_across_horizons(score, target))
    for drop in RISK_OFF:
        off_minus = [s for s in RISK_OFF if s != drop]
        score = build_roo_score(rw, RISK_ON, off_minus)
        rows.append((f"OFF drop {drop}",) + _ic_across_horizons(score, target))
    _print_table("(3) Leave-one-out ablation (formation = 5d) — which member fixes IC if removed?", rows)

    # 4. Restructure: predict equities from non-equity risk-on (no target leak).
    rows = []
    on_no_us_eq = [s for s in RISK_ON if s not in ("SPY", "QQQ", "IWM")]
    for label, ret in (("5d", rw), ("21d", rm), ("63d", rq)):
        score = build_roo_score(ret, on_no_us_eq, RISK_OFF)
        rows.append((f"ON\\(SPY,QQQ,IWM) — formation={label}",) + _ic_across_horizons(score, target))
    _print_table("(4) Drop US equities from ON basket entirely (predict-not-include)", rows)

    # 5. Alternative target: SPY−TLT spread forward log-return. This is what the
    #    ROO score actually measures the rotation INTO — not SPY level alone.
    if {"SPY", "TLT"}.issubset(close.columns):
        rows = []
        spread = np.log(close["SPY"] / close["TLT"])
        spread_target = lambda h: spread.shift(-h) - spread
        for label, ret in (("5d", rw), ("21d", rm), ("63d", rq)):
            score = build_roo_score(ret, RISK_ON, RISK_OFF)
            r5 = _ic_summary(score, spread_target(5))
            r21 = _ic_summary(score, spread_target(21))
            r63 = _ic_summary(score, spread_target(63))
            h5 = _hit_rate(score, spread_target(5))
            h21 = _hit_rate(score, spread_target(21))
            h63 = _hit_rate(score, spread_target(63))
            rows.append((f"headline (formation={label}) → SPY−TLT spread",
                         r5["spearman"], h5["hit"], r21["spearman"], h21["hit"],
                         r63["spearman"], h63["hit"], r5["n"]))
        _print_table("(5) Headline ROO predicting the SPY−TLT spread (the natural target)", rows)


if __name__ == "__main__":
    main()
