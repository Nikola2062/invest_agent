"""Position sizing — convert ratings + market data into target weights.

The agent stack produces a *rating* per ticker (Buy / Overweight / Hold /
Underweight / Sell). That rating is a directional signal, not a position
size. This module is where the rating becomes a number you'd actually
trade.

Two sizers:

  ``equal_weight_sizer``
    Long-only, equal weight across names rated Buy or Overweight. The
    Phase 2 default. Used as the "naive" baseline.

  ``risk_aware_sizer``
    Rating → signed multiplier → inverse-volatility scaling → per-name
    cap → per-sector cap → gross-exposure cap. The realistic policy.

Both implement the ``Sizer`` protocol so callers (the backtest engine,
live trading, ad-hoc analysis) can swap them freely.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Optional, Protocol

import numpy as np
import pandas as pd

from tradingagents.backtest.strategy import Decision

logger = logging.getLogger(__name__)


# Rating → signed multiplier on the base weight. Buy is twice an
# Overweight; Sell is twice an Underweight. Hold is flat.
_DEFAULT_MULTIPLIERS: dict[str, float] = {
    "Buy": 2.0,
    "Overweight": 1.0,
    "Hold": 0.0,
    "Underweight": -1.0,
    "Sell": -2.0,
}


@dataclass
class SizingConfig:
    """Knobs for ``risk_aware_sizer``.

    Defaults are deliberately mid-of-the-road for a long-only mid-cap
    research portfolio: ~10% per-name cap, ~30% per-sector cap, 100%
    gross. Tighten for a less concentrated policy; loosen for a more
    aggressive one.
    """

    # Rating handling
    base_weight: float = 0.05
    rating_multipliers: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_MULTIPLIERS)
    )
    allow_shorts: bool = False         # if False, negative weights are zeroed

    # Vol scaling
    use_vol_scaling: bool = True
    target_vol: float = 0.20           # ann. portfolio vol target (informational)
    vol_lookback_days: int = 60        # trailing window for realized vol
    vol_floor: float = 0.05            # min realized vol; avoids 1/0 blowup

    # Risk caps
    max_position: float = 0.10         # |w_i| ≤ 10% per name
    max_sector_exposure: float = 0.30  # sum(|w_i| in sector) ≤ 30%
    max_gross: float = 1.0             # sum(|w_i|) ≤ 100%


class Sizer(Protocol):
    """Callable converting decisions + context into a {ticker: weight} dict.

    ``prices`` is optional because the equal-weight sizer doesn't need it.
    ``as_of`` is required for any vol/lookback computation so the sizer
    only uses information knowable at the rebalance date.
    """

    def __call__(
        self,
        decisions: Iterable[Decision],
        *,
        prices: Optional[pd.DataFrame] = None,
        sectors: Optional[Mapping[str, str]] = None,
        as_of: Optional[pd.Timestamp] = None,
        config: Optional[SizingConfig] = None,
    ) -> dict[str, float]: ...


# --- equal-weight (baseline) ----------------------------------------------


_INCLUDE_LONG = {"Buy", "Overweight"}


def equal_weight_sizer(
    decisions: Iterable[Decision],
    *,
    prices: Optional[pd.DataFrame] = None,  # noqa: ARG001 — preserved for protocol parity
    sectors: Optional[Mapping[str, str]] = None,  # noqa: ARG001
    as_of: Optional[pd.Timestamp] = None,  # noqa: ARG001
    config: Optional[SizingConfig] = None,  # noqa: ARG001
) -> dict[str, float]:
    """Long-only, equal weight over Buy/Overweight names.

    Preserves the Phase 2 default. Kept as a named sizer (rather than
    inlined) so the backtester can A/B against the risk-aware version
    without code surgery.
    """
    longs = [d.ticker for d in decisions if d.rating in _INCLUDE_LONG]
    if not longs:
        return {}
    w = 1.0 / len(longs)
    return {t: w for t in longs}


# --- risk-aware sizer -----------------------------------------------------


def _realised_vol(prices: pd.DataFrame, ticker: str, as_of: pd.Timestamp, lookback: int) -> float | None:
    """Annualised realised vol from daily-close returns up to ``as_of``.

    Returns None when there isn't enough history (newly listed, missing
    prices). Callers fall back to a config floor.
    """
    if ticker not in prices.columns:
        return None
    series = prices[ticker].loc[:as_of].dropna()
    if len(series) < lookback + 1:
        return None
    rets = series.tail(lookback + 1).pct_change().dropna()
    if rets.empty:
        return None
    return float(rets.std(ddof=1) * np.sqrt(252))


def _apply_position_cap(weights: dict[str, float], cap: float) -> dict[str, float]:
    """Clip each |w_i| at the per-name cap, preserving sign."""
    return {t: float(np.sign(w) * min(abs(w), cap)) for t, w in weights.items()}


def _apply_sector_cap(
    weights: dict[str, float],
    sectors: Mapping[str, str],
    cap: float,
) -> dict[str, float]:
    """Proportionally shrink within each sector if its exposure exceeds the cap.

    Names without a sector entry land in ``"Unknown"`` and share that
    bucket's budget — better than silently bypassing the cap.
    """
    by_sector: dict[str, list[str]] = defaultdict(list)
    for ticker in weights:
        by_sector[sectors.get(ticker, "Unknown")].append(ticker)

    scaled = dict(weights)
    for _sector, tickers in by_sector.items():
        exposure = sum(abs(scaled[t]) for t in tickers)
        if exposure > cap and exposure > 0:
            factor = cap / exposure
            for t in tickers:
                scaled[t] = scaled[t] * factor
    return scaled


def _apply_gross_cap(weights: dict[str, float], cap: float) -> dict[str, float]:
    """Proportionally shrink the whole book to satisfy a gross-exposure ceiling."""
    gross = sum(abs(w) for w in weights.values())
    if gross <= cap or gross == 0:
        return weights
    factor = cap / gross
    return {t: w * factor for t, w in weights.items()}


def risk_aware_sizer(
    decisions: Iterable[Decision],
    *,
    prices: Optional[pd.DataFrame] = None,
    sectors: Optional[Mapping[str, str]] = None,
    as_of: Optional[pd.Timestamp] = None,
    config: Optional[SizingConfig] = None,
) -> dict[str, float]:
    """Rating → multiplier → vol-scale → per-name cap → sector cap → gross cap.

    Each stage is monotone in the sense that it can only shrink exposures,
    never grow them, so the final result strictly satisfies every cap. Vol
    scaling normalises target-vol contribution across names; without it,
    a 60% vol biotech and a 15% vol utility would carry the same risk
    weight, which is rarely what a trader actually wants.

    Missing data is handled deliberately:
      - No price history → vol scaling skipped for that name (uses base weight).
      - No sector entry → bucketed under ``"Unknown"``; sector cap still applies.
      - Empty decision list → empty result (portfolio in cash).
    """
    cfg = config or SizingConfig()
    sectors = sectors or {}
    multipliers = cfg.rating_multipliers

    # Step 1: rating → signed base weight
    raw: dict[str, float] = {}
    for d in decisions:
        mult = multipliers.get(d.rating, 0.0)
        if not cfg.allow_shorts and mult < 0:
            mult = 0.0
        if mult == 0:
            continue
        raw[d.ticker] = mult * cfg.base_weight

    if not raw:
        return {}

    # Step 2: inverse-volatility scaling
    if cfg.use_vol_scaling and prices is not None and as_of is not None:
        vols = {
            t: _realised_vol(prices, t, pd.Timestamp(as_of), cfg.vol_lookback_days)
            for t in raw
        }
        # Scale each weight by target_vol / max(vol, floor). Names with
        # missing vol skip the scaling (use the base multiplier instead
        # of dropping them — dropping would silently bias toward
        # long-history names).
        for t in list(raw):
            v = vols.get(t)
            if v is None:
                continue
            v = max(v, cfg.vol_floor)
            raw[t] *= cfg.target_vol / v

    # Step 3: per-name cap
    capped = _apply_position_cap(raw, cfg.max_position)

    # Step 4: per-sector cap (only meaningful when sectors are provided)
    if sectors:
        capped = _apply_sector_cap(capped, sectors, cfg.max_sector_exposure)

    # Step 5: gross-exposure cap (the final backstop)
    final = _apply_gross_cap(capped, cfg.max_gross)

    # Drop near-zero weights so the equity curve doesn't carry dust.
    return {t: w for t, w in final.items() if abs(w) > 1e-9}
