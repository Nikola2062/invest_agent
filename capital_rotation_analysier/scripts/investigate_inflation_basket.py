"""Per-constituent dissection of the inflation_score basket.

Hypotheses under test:
  (A) Target leakage — CPER is in the basket (weight +0.25) AND is the
      FORWARD_ASSET target. Score includes recent CPER returns; testing it
      against forward CPER returns asks "does last month's copper predict next
      month's copper?" At monthly horizons commodities mean-revert, so this is
      structurally anti-predictive. Removing CPER from the basket and/or
      changing the target should fix it.
  (B) Wrong target — inflation is a slow macro variable; its natural target
      is not next-month commodity returns but inflation-sensitive *rotations*:
      e.g. (CPER+USO) − TLT forward spread, or HYG−LQD credit spread (credit
      widens when inflation surprises hit risk assets), or a broad
      commodities basket forward.
  (C) TLT's contribution is noisy — TLT has duration risk unrelated to
      inflation (Fed-pause bid, term premia moves). Its negative weight may
      be subtracting more noise than signal. Test by zeroing/inverting.
  (D) Sign on a member is wrong — try flipping each input's weight and
      checking if any single flip fixes IC.

Run from project root:
  PYTHONPATH=src .venv/bin/python scripts/investigate_inflation_basket.py
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


# Current §3.2.5 weights — copied from signals.inflation_score
WEIGHTS = {
    "CPER":  0.25,
    "USO":   0.25,
    "GLD":   0.10,
    "SLV":   0.10,
    "TLT":  -0.20,
}

# Symbols we'll need across the whole script.
ALL_SYMBOLS = list(WEIGHTS.keys()) + ["SPY", "IEF", "HYG", "LQD"]


def _load_close_panel(con, symbols: list[str]) -> pd.DataFrame:
    df = con.execute(
        "SELECT symbol, ts, adj_close FROM raw_bars "
        "WHERE symbol IN ({}) AND adj_close IS NOT NULL "
        "AND asset_class != 'equity_hk' "
        "ORDER BY ts".format(",".join(f"'{s}'" for s in symbols))
    ).df()
    df["ts"] = pd.to_datetime(df["ts"])
    panel = df.pivot(index="ts", columns="symbol", values="adj_close").sort_index()
    nyse = mcal.get_calendar("NYSE")
    sched = nyse.schedule(start_date=panel.index.min(), end_date=panel.index.max())
    nyse_days = pd.DatetimeIndex(pd.to_datetime(sched.index).normalize())
    return panel.loc[panel.index.intersection(nyse_days)]


def _fwd_log_return(close: pd.Series, h: int) -> pd.Series:
    return np.log(close.shift(-h) / close)


def _ic_summary(score: pd.Series, fwd: pd.Series) -> tuple[float | None, int]:
    s = pd.concat([score.rename("s"), fwd.rename("f")], axis=1).dropna()
    if len(s) < 30:
        return (None, int(len(s)))
    return (float(s["s"].rank().corr(s["f"].rank())), int(len(s)))


def _hit(score: pd.Series, fwd: pd.Series) -> float | None:
    s = pd.concat([score.rename("s"), fwd.rename("f")], axis=1).dropna()
    s = s[(s["s"].abs() > 0) & (s["f"].abs() > 0)]
    if len(s) < 30:
        return None
    return float((np.sign(s["s"]) == np.sign(s["f"])).mean())


def _across(score: pd.Series, target: pd.Series) -> tuple:
    out = []
    for h in (5, 21, 63):
        f = _fwd_log_return(target, h)
        m, n = _ic_summary(score, f)
        out.append(m)
        out.append(_hit(score, f))
        if h == 5:
            n_obs = n
    out.append(n_obs)
    return tuple(out)


def _print(title: str, rows: list[tuple]) -> None:
    print(f"\n### {title}\n")
    print("| component | 5d IC | 5d hit | 21d IC | 21d hit | 63d IC | 63d hit | n |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    fmt = lambda x: f"{x:+.3f}" if x is not None else "—"
    fmth = lambda x: f"{x*100:.0f}%" if x is not None else "—"
    for r in rows:
        name, m5, h5, m21, h21, m63, h63, n = r
        print(f"| {name} | {fmt(m5)} | {fmth(h5)} | {fmt(m21)} | "
              f"{fmth(h21)} | {fmt(m63)} | {fmth(h63)} | {n} |")


def _build_inflation(returns: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Construct the headline inflation score series from any weight scheme."""
    z = returns.apply(lambda c: M.robust_z(c), axis=0)
    weighted = pd.Series(0.0, index=returns.index)
    used = 0.0
    for sym, w in weights.items():
        if sym not in z.columns:
            continue
        weighted = weighted.add(z[sym] * w, fill_value=0.0)
        used += abs(w)
    if used < 0.5:
        return pd.Series(dtype=float)
    return np.tanh(weighted / 2.0) * 100.0


def main() -> None:
    cfg = load_config(Path(__file__).resolve().parent.parent / "config.yaml")
    print("# inflation basket dissection")
    print()
    print("Each row tests one component (signed by its current weight) or "
          "a structural variant. Targets cover the natural candidates: CPER "
          "(current FORWARD_ASSET), USO, TLT, plus inflation-sensitive *rotation* "
          "spreads (CPER−TLT, HYG−LQD).")
    print()
    print(f"- Current weights: {WEIGHTS}")
    print(f"- Current formation: 21d (already lengthened)")

    with connect(cfg.storage.duckdb_path) as con:
        close = _load_close_panel(con, ALL_SYMBOLS)

    print(f"\n_Bars loaded: {close.shape[0]} sessions, "
          f"{close.shape[1]} symbols. Range: "
          f"{close.index.min().date()} → {close.index.max().date()}._")

    rm = M.log_returns(close, 21)
    rw = M.log_returns(close, 5)
    rq = M.log_returns(close, 63)

    if "CPER" not in close.columns:
        print("\n_CPER missing — cannot reproduce harness target._")
        return

    cper = close["CPER"]

    # (1) Per-constituent IC at 21d formation, signed by current weight.
    rows = []
    for sym, w in WEIGHTS.items():
        if sym not in close.columns:
            continue
        z = M.robust_z(rm[sym]) * np.sign(w)  # sign by weight; magnitude is uniform 1
        rows.append((f"{sym} (w={w:+.2f}) → CPER",) + _across(z, cper))
    _print("(1) Per-constituent IC vs forward CPER — formation = r_m (21d), signed by weight", rows)

    # (2) Headline at different formation horizons. Already 21d in production;
    #     verify whether 5d or 63d changes anything (we expect not, since
    #     commodities don't trend at monthly frequencies).
    rows = []
    for label, ret in (("5d", rw), ("21d", rm), ("63d", rq)):
        score = _build_inflation(ret, WEIGHTS)
        rows.append((f"headline (formation={label}) → CPER",) + _across(score, cper))
    _print("(2) Headline at different formation horizons vs CPER", rows)

    # (3) Leave-one-out ablation at 21d formation against CPER.
    rows = []
    for drop in WEIGHTS:
        if drop not in close.columns:
            continue
        w2 = {k: v for k, v in WEIGHTS.items() if k != drop}
        score = _build_inflation(rm, w2)
        rows.append((f"drop {drop} → CPER",) + _across(score, cper))
    _print("(3) Leave-one-out (21d formation) — which member fixes IC if removed?", rows)

    # (4) Sign-flip each constituent (suspect: wrong direction on a member).
    rows = []
    for flip in WEIGHTS:
        if flip not in close.columns:
            continue
        w2 = dict(WEIGHTS)
        w2[flip] = -w2[flip]
        score = _build_inflation(rm, w2)
        rows.append((f"flip sign on {flip} → CPER",) + _across(score, cper))
    _print("(4) Sign-flip ablation (21d formation, current basket members)", rows)

    # (5) Test against ALTERNATE targets with the current headline. If the
    #     score is well-formed but the test asset is wrong, this should reveal it.
    score = _build_inflation(rm, WEIGHTS)
    rows = []
    if "USO" in close.columns:
        rows.append(("→ USO",) + _across(score, close["USO"]))
    if "TLT" in close.columns:
        rows.append(("→ TLT",) + _across(score, close["TLT"]))
        rows.append(("→ −TLT (inflation breakout)",) + _across(score, -np.log(close["TLT"])))
    if {"CPER", "TLT"}.issubset(close.columns):
        spread = np.log(close["CPER"] / close["TLT"])
        rows.append(("→ CPER−TLT spread (inflation rotation)",) + _across(score, spread))
    if {"USO", "TLT"}.issubset(close.columns):
        spread = np.log(close["USO"] / close["TLT"])
        rows.append(("→ USO−TLT spread",) + _across(score, spread))
    if {"HYG", "LQD"}.issubset(close.columns):
        spread = np.log(close["HYG"] / close["LQD"])
        rows.append(("→ HYG−LQD credit (widens with inflation surprises)",) + _across(score, spread))
    # Broad-commodity forward: equal-weighted basket
    cmdty_syms = [s for s in ("CPER", "USO", "GLD", "SLV") if s in close.columns]
    if cmdty_syms:
        # Geometric mean log price → equal-weight log forward return
        broad = np.log(close[cmdty_syms]).mean(axis=1)
        rows.append(("→ broad commodity basket (CPER/USO/GLD/SLV mean)",) + _across(score, broad))
    _print("(5) Headline (current basket, 21d formation) vs alternate forward targets", rows)

    # (6) Drop CPER from the basket (target-leak fix) and re-test vs CPER.
    weights_no_cper = {k: v for k, v in WEIGHTS.items() if k != "CPER"}
    score_nl = _build_inflation(rm, weights_no_cper)
    rows = []
    rows.append(("CPER-free score → CPER",) + _across(score_nl, cper))
    if "USO" in close.columns:
        rows.append(("CPER-free score → USO",) + _across(score_nl, close["USO"]))
    if {"CPER", "TLT"}.issubset(close.columns):
        spread = np.log(close["CPER"] / close["TLT"])
        rows.append(("CPER-free score → CPER−TLT spread",) + _across(score_nl, spread))
    if cmdty_syms:
        broad = np.log(close[cmdty_syms]).mean(axis=1)
        rows.append(("CPER-free score → broad commodity basket",) + _across(score_nl, broad))
    _print("(6) Drop CPER from basket (remove target leak) and re-test", rows)


if __name__ == "__main__":
    main()
