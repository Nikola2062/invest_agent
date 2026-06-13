"""Tests for the Relative Strength Agent — ticker vs benchmark return ratios."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.agents.relative_strength import assess_relative_strength
from src.models.schemas import RelativeStrengthCfg


def test_strong_leader_when_ticker_dominates():
    """Ticker up 60% / 200%, sector up 10% / 30%, index up 5% / 15% — strong outperformer."""
    cfg = RelativeStrengthCfg(windows_days=[90, 365], benchmarks={
        "US": {"market_index": "SPY", "sector_etfs": {"Technology": "IGV"}},
    })

    # Stub the benchmark fetcher so the test doesn't hit yfinance
    from src.agents import relative_strength as mod
    mod.fetch_returns = lambda sym, windows: {
        90: {"IGV": 10.0, "SPY": 5.0}.get(sym),
        365: {"IGV": 30.0, "SPY": 15.0}.get(sym),
    }
    out = assess_relative_strength(
        symbol="FAKE", market="US", sector="Technology",
        ticker_returns={90: 60.0, 365: 200.0},
        cfg=cfg,
    )
    assert out.signal in ("strong_leader", "leader"), f"got {out.signal}"
    assert out.vs_sector_etf_90d is not None and out.vs_sector_etf_90d > 1.2
    print(f"✓ Outperformer → {out.signal} (vs sector 90d {out.vs_sector_etf_90d}, 365d {out.vs_sector_etf_365d})")


def test_weak_laggard_when_ticker_underperforms():
    cfg = RelativeStrengthCfg(windows_days=[90, 365], benchmarks={
        "US": {"market_index": "SPY", "sector_etfs": {"Technology": "IGV"}},
    })
    from src.agents import relative_strength as mod
    mod.fetch_returns = lambda sym, windows: {
        90: {"IGV": 20.0, "SPY": 10.0}.get(sym),
        365: {"IGV": 50.0, "SPY": 20.0}.get(sym),
    }
    out = assess_relative_strength(
        symbol="FAKE", market="US", sector="Technology",
        ticker_returns={90: -20.0, 365: -40.0},
        cfg=cfg,
    )
    assert out.signal in ("laggard", "weak_laggard"), f"got {out.signal}"
    print(f"✓ Underperformer → {out.signal}")


def test_neutral_when_no_benchmark_data():
    cfg = RelativeStrengthCfg(windows_days=[90, 365], benchmarks={
        "US": {"market_index": "SPY", "sector_etfs": {"Technology": "IGV"}},
    })
    from src.agents import relative_strength as mod
    mod.fetch_returns = lambda sym, windows: {90: None, 365: None}
    out = assess_relative_strength(
        symbol="FAKE", market="US", sector="Technology",
        ticker_returns={90: 30.0, 365: 80.0},
        cfg=cfg,
    )
    assert out.signal == "neutral"
    assert out.vs_sector_etf_90d is None
    print("✓ No benchmark data → neutral signal, ratios None")
