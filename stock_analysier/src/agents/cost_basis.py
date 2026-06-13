"""Cost-Basis Agent — volume-by-price histogram (stage 3 of the investment framework).

Builds a price-volume histogram over the lookback window. Each bucket holds the
total volume traded while the typical price (= (H+L+C)/3) sat inside it. Buckets
containing ≥ hvn_min_volume_pct of the total window volume are flagged as
High-Volume Nodes (HVNs) — these mark the strongest support / resistance zones.

Trapped-supply % = total volume sitting above current price by more than 10%.
Accumulation %   = total volume sitting below current price within 10%.

No LLM. Receives only OHLCV bars + config. Blinded from the fundamental side.
"""
from __future__ import annotations

import logging
from typing import Literal

import numpy as np
import pandas as pd

from src.models.schemas import CostBasisLevel, CostBasisMap, CostBasisCfg

log = logging.getLogger(__name__)


def _merge_adjacent_buckets(
    buckets: list[tuple[float, float, float]],
    bucket_width_pct: float,
) -> list[tuple[float, float, float]]:
    """Coalesce contiguous high-volume buckets into a single level. Returns
    (price_low, price_high, volume_pct) for each merged level.
    """
    if not buckets:
        return []
    buckets = sorted(buckets, key=lambda b: b[0])
    merged: list[list[float]] = []
    for lo, hi, vol in buckets:
        if merged and lo <= merged[-1][1] * (1.0 + bucket_width_pct / 100.0):
            merged[-1][1] = max(merged[-1][1], hi)
            merged[-1][2] += vol
        else:
            merged.append([lo, hi, vol])
    return [(m[0], m[1], m[2]) for m in merged]


def assess_cost_basis(
    bars: pd.DataFrame,
    cfg: CostBasisCfg,
    current_price: float,
) -> CostBasisMap:
    if bars is None or bars.empty or "Volume" not in bars.columns or len(bars) < 30:
        return CostBasisMap(
            lookback_days=cfg.lookback_days,
            hvn_levels=[],
            trapped_supply_pct=0.0,
            accumulation_pct=0.0,
            summary="insufficient_history",
        )

    window = bars.tail(cfg.lookback_days)
    typical = ((window["High"] + window["Low"] + window["Close"]) / 3.0).astype(float)
    vol = window["Volume"].astype(float).fillna(0.0)

    total_vol = float(vol.sum())
    if total_vol <= 0:
        return CostBasisMap(
            lookback_days=cfg.lookback_days, hvn_levels=[],
            trapped_supply_pct=0.0, accumulation_pct=0.0,
            summary="no_volume_in_window",
        )

    # Build buckets — geometric, bucket_pct_width % wide each
    p_min = float(typical.min())
    p_max = float(typical.max())
    if p_min <= 0 or p_max <= p_min:
        return CostBasisMap(
            lookback_days=cfg.lookback_days, hvn_levels=[],
            trapped_supply_pct=0.0, accumulation_pct=0.0,
            summary="degenerate_price_range",
        )

    step = cfg.bucket_pct_width / 100.0
    edges: list[float] = []
    e = p_min
    while e < p_max * (1.0 + step):
        edges.append(e)
        e *= (1.0 + step)
    if edges[-1] < p_max:
        edges.append(p_max * (1.0 + step))

    # Aggregate volume per bucket
    bucket_vol = np.zeros(len(edges) - 1)
    for price, v in zip(typical.values, vol.values):
        idx = int(np.searchsorted(edges, price, side="right") - 1)
        idx = max(0, min(idx, len(bucket_vol) - 1))
        bucket_vol[idx] += v

    # HVN detection: any bucket holding ≥ hvn_min_volume_pct of total
    hvn_threshold_vol = total_vol * (cfg.hvn_min_volume_pct / 100.0)
    raw_hvns: list[tuple[float, float, float]] = []
    for i, v in enumerate(bucket_vol):
        if v >= hvn_threshold_vol:
            raw_hvns.append((edges[i], edges[i + 1], v / total_vol * 100.0))
    merged = _merge_adjacent_buckets(raw_hvns, cfg.bucket_pct_width)

    # Build CostBasisLevel entries
    levels: list[CostBasisLevel] = []
    for lo, hi, pct in merged:
        if current_price < lo:
            position = "above"
            # above current → potential resistance / trapped supply
            role: Literal["support", "resistance", "neutral"] = "resistance"
        elif current_price > hi:
            position = "below"
            # below current → potential support / accumulation zone
            role = "support"
        else:
            position = "at"
            role = "neutral"
        levels.append(CostBasisLevel(
            price_low=round(lo, 2), price_high=round(hi, 2),
            volume_pct_of_window=round(pct, 1),
            position_vs_current=position,
            role=role,
        ))

    # Trapped-supply: volume in buckets entirely above current * 1.10
    threshold_above = current_price * 1.10
    threshold_below = current_price * 0.90
    trapped_vol = 0.0
    accum_vol = 0.0
    for i, v in enumerate(bucket_vol):
        bucket_mid = (edges[i] + edges[i + 1]) / 2.0
        if bucket_mid >= threshold_above:
            trapped_vol += v
        elif threshold_below <= bucket_mid < current_price:
            accum_vol += v
    trapped_pct = trapped_vol / total_vol * 100.0
    accum_pct = accum_vol / total_vol * 100.0

    summary = (
        f"{len(levels)} HVN level(s) over {cfg.lookback_days}d; "
        f"trapped supply {trapped_pct:.0f}%, accumulation {accum_pct:.0f}%"
    )

    return CostBasisMap(
        lookback_days=cfg.lookback_days,
        hvn_levels=levels,
        trapped_supply_pct=round(trapped_pct, 1),
        accumulation_pct=round(accum_pct, 1),
        summary=summary,
    )
