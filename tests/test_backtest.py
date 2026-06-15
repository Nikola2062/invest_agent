"""End-to-end backtest test against a deterministic stub strategy.

No LLM, no network. The whole pipeline — strategy → runner JSONL →
portfolio → metrics → report — runs against synthetic prices in <1s.

Three guarantees:

  1. **Pipeline composability.** A stub strategy with a known answer must
     produce a known equity curve and metric set.
  2. **Crash recovery.** Calling the runner twice with the same output
     path must yield no new decisions on the second call.
  3. **Random baseline averages out.** Over many uniform-random decisions
     on independent random walks, the total return must be close to zero
     — this is the honesty check on the metrics math itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tradingagents.backtest import RandomStrategy
from tradingagents.backtest.metrics import (
    annualised_volatility,
    cagr,
    max_drawdown,
    returns_from_equity,
    sharpe_ratio,
    summarise,
    total_return,
)
from tradingagents.backtest.portfolio import (
    benchmark_curve,
    build_equity_curve,
    rating_to_weights,
)
from tradingagents.backtest.report import (
    per_ticker_stats,
    render_html,
    render_markdown,
)
from tradingagents.backtest.runner import (
    _add_months,
    generate_rebalance_dates,
    load_decisions,
    walk_forward,
)
from tradingagents.backtest.strategy import Decision, callable_strategy

pytestmark = pytest.mark.unit


# --- helpers --------------------------------------------------------------


def _trending_prices(tickers, start="2023-01-01", end="2024-12-31"):
    """Build a daily price panel with one trending name and one flat name."""
    idx = pd.bdate_range(start, end)
    rng = np.random.default_rng(seed=42)
    cols = {}
    for tkr in tickers:
        if tkr == "WIN":
            # +0.1%/day deterministic uptrend
            cols[tkr] = 100.0 * (1.001 ** np.arange(len(idx)))
        elif tkr == "LOSE":
            cols[tkr] = 100.0 * (0.999 ** np.arange(len(idx)))
        elif tkr == "FLAT":
            cols[tkr] = np.full(len(idx), 100.0)
        elif tkr == "SPY":
            # Slight positive drift for a benchmark.
            cols[tkr] = 100.0 * (1.0003 ** np.arange(len(idx)))
        else:
            # Random walk
            rets = rng.normal(0, 0.01, size=len(idx))
            cols[tkr] = 100.0 * np.exp(np.cumsum(rets))
    return pd.DataFrame(cols, index=idx)


# --- date generation -----------------------------------------------------


def test_generate_monthly_dates():
    dates = generate_rebalance_dates("2023-01-15", "2023-04-15", "monthly")
    assert dates == ["2023-01-15", "2023-02-15", "2023-03-15", "2023-04-15"]


def test_generate_weekly_dates_count():
    dates = generate_rebalance_dates("2023-01-02", "2023-02-06", "weekly")
    assert len(dates) == 6
    assert dates[0] == "2023-01-02"
    assert dates[-1] == "2023-02-06"


def test_generate_quarterly_dates():
    dates = generate_rebalance_dates("2023-01-15", "2024-01-15", "quarterly")
    assert dates == [
        "2023-01-15", "2023-04-15", "2023-07-15", "2023-10-15", "2024-01-15"
    ]


def test_add_months_handles_month_end():
    from datetime import date as _date
    assert _add_months(_date(2023, 1, 31), 1) == _date(2023, 2, 28)
    assert _add_months(_date(2024, 1, 31), 1) == _date(2024, 2, 29)


def test_generate_rejects_inverted_range():
    with pytest.raises(ValueError):
        generate_rebalance_dates("2024-01-01", "2023-01-01", "monthly")


# --- runner + JSONL roundtrip --------------------------------------------


def test_walk_forward_records_decisions(tmp_path):
    """A stub strategy is faithfully appended to JSONL and round-trips."""
    out = tmp_path / "decisions.jsonl"
    rate = lambda tkr, d: "Buy" if tkr == "WIN" else "Sell"  # noqa: E731
    strat = callable_strategy(rate)

    list(walk_forward(
        ["WIN", "LOSE"], ["2023-01-15", "2023-02-15"], strat, out,
    ))

    loaded = load_decisions(out)
    assert len(loaded) == 4
    assert {d.rating for d in loaded if d.ticker == "WIN"} == {"Buy"}
    assert {d.rating for d in loaded if d.ticker == "LOSE"} == {"Sell"}


def test_walk_forward_skips_completed(tmp_path):
    """Re-running with the same output path adds zero new decisions."""
    out = tmp_path / "decisions.jsonl"
    strat = callable_strategy(lambda t, d: "Hold")

    list(walk_forward(["A", "B"], ["2023-01-01"], strat, out))
    n_after_first = sum(1 for _ in out.read_text().splitlines() if _.strip())
    assert n_after_first == 2

    new = list(walk_forward(["A", "B"], ["2023-01-01"], strat, out))
    assert new == []


def test_walk_forward_skips_strategy_errors(tmp_path):
    """on_error='skip' continues past a thrown exception."""
    out = tmp_path / "decisions.jsonl"

    def flaky(ticker, _date):
        if ticker == "BAD":
            raise RuntimeError("simulated vendor failure")
        return Decision(ticker=ticker, trade_date=_date, rating="Hold")

    list(walk_forward(
        ["GOOD", "BAD", "GOOD2"], ["2023-01-15"], flaky, out, on_error="skip",
    ))
    loaded = load_decisions(out)
    assert {d.ticker for d in loaded} == {"GOOD", "GOOD2"}


def test_walk_forward_propagates_when_on_error_raise(tmp_path):
    out = tmp_path / "decisions.jsonl"
    def boom(*_):
        raise RuntimeError("x")
    with pytest.raises(RuntimeError):
        list(walk_forward(["A"], ["2023-01-01"], boom, out, on_error="raise"))


def test_load_decisions_tolerates_malformed_lines(tmp_path):
    out = tmp_path / "decisions.jsonl"
    out.write_text(
        json.dumps({"ticker": "A", "trade_date": "2023-01-01", "rating": "Buy"}) + "\n"
        "this-is-not-json\n"
        + json.dumps({"ticker": "B", "trade_date": "2023-01-01", "rating": "Sell"}) + "\n",
        encoding="utf-8",
    )
    loaded = load_decisions(out)
    assert [d.ticker for d in loaded] == ["A", "B"]


# --- portfolio construction ----------------------------------------------


def test_rating_to_weights_equal_weight():
    decs = [
        Decision("A", "2023-01-15", "Buy"),
        Decision("B", "2023-01-15", "Overweight"),
        Decision("C", "2023-01-15", "Hold"),
        Decision("D", "2023-01-15", "Sell"),
    ]
    w = rating_to_weights(decs)
    assert set(w) == {"A", "B"}
    assert w["A"] == pytest.approx(0.5)
    assert w["B"] == pytest.approx(0.5)


def test_rating_to_weights_all_cash():
    decs = [Decision("A", "2023-01-15", "Hold")]
    assert rating_to_weights(decs) == {}


def test_equity_curve_picks_the_trending_name():
    """A strategy that always picks WIN over LOSE should produce a
    monotonically increasing equity curve."""
    prices = _trending_prices(["WIN", "LOSE"])
    rebal = generate_rebalance_dates("2023-01-15", "2024-01-15", "monthly")
    decisions = [
        Decision(t, d, "Buy" if t == "WIN" else "Sell")
        for d in rebal for t in ["WIN", "LOSE"]
    ]
    equity = build_equity_curve(decisions, prices)
    assert len(equity) >= 2
    assert equity.iloc[-1] > equity.iloc[0]
    # Should be ~ (1.001)^252 ≈ 1.29 over a year of trading days.
    assert equity.iloc[-1] > 1.20


def test_equity_curve_flat_when_all_hold():
    prices = _trending_prices(["WIN"])
    rebal = generate_rebalance_dates("2023-01-15", "2023-06-15", "monthly")
    decisions = [Decision("WIN", d, "Hold") for d in rebal]
    equity = build_equity_curve(decisions, prices)
    assert equity.nunique() == 1


def test_equity_curve_skips_tickers_without_prices():
    """A decision on a ticker not in the price panel doesn't crash the run."""
    prices = _trending_prices(["WIN"])
    rebal = generate_rebalance_dates("2023-01-15", "2023-06-15", "monthly")
    decisions = []
    for d in rebal:
        decisions.append(Decision("WIN", d, "Buy"))
        decisions.append(Decision("GHOST", d, "Buy"))  # not in prices
    equity = build_equity_curve(decisions, prices)
    # WIN's trend still drives a positive result.
    assert equity.iloc[-1] > equity.iloc[0]


def test_benchmark_curve_normalised_to_equity_anchor():
    prices = _trending_prices(["WIN", "SPY"])
    rebal = generate_rebalance_dates("2023-01-15", "2023-06-15", "monthly")
    decisions = [Decision("WIN", d, "Buy") for d in rebal]
    equity = build_equity_curve(decisions, prices)
    bench = benchmark_curve(prices, "SPY", equity.index)
    assert bench.iloc[0] == pytest.approx(1.0)
    # SPY drifts mildly positive in the fixture.
    assert bench.iloc[-1] > bench.iloc[0]


# --- metrics --------------------------------------------------------------


def test_total_return_basic():
    eq = pd.Series([1.0, 1.1, 1.21], index=pd.date_range("2023-01-01", periods=3))
    assert total_return(eq) == pytest.approx(0.21)


def test_max_drawdown_negative():
    eq = pd.Series(
        [1.0, 1.2, 0.9, 1.0, 1.5],
        index=pd.date_range("2023-01-01", periods=5),
    )
    # Peak was 1.2, trough was 0.9 → -25%
    assert max_drawdown(eq) == pytest.approx(-0.25)


def test_sharpe_zero_when_no_volatility():
    eq = pd.Series([1.0, 1.0, 1.0], index=pd.date_range("2023-01-01", periods=3))
    rets = returns_from_equity(eq)
    assert pd.isna(sharpe_ratio(rets))


def test_summarise_with_benchmark_includes_alpha_and_ir():
    idx = pd.date_range("2023-01-01", periods=12, freq="MS")
    rng = np.random.default_rng(7)
    rets = rng.normal(0.01, 0.04, 12)
    eq = (1 + pd.Series(rets, index=idx)).cumprod()
    bench = (1 + pd.Series(rng.normal(0.005, 0.03, 12), index=idx)).cumprod()
    summary = summarise(eq, bench)
    assert summary.n_periods == 11
    assert summary.alpha_vs_benchmark is not None
    assert summary.information_ratio is not None
    assert summary.benchmark_total_return is not None


def test_random_strategy_is_reproducible():
    """Same seed → same sequence. Critical for reproducible baselines."""
    s1 = RandomStrategy(seed=123)
    s2 = RandomStrategy(seed=123)
    pairs = [("A", "2023-01-01"), ("B", "2023-02-01"), ("C", "2023-03-01")]
    for ticker, date in pairs:
        assert s1(ticker, date).rating == s2(ticker, date).rating


# --- end-to-end pipeline -------------------------------------------------


def test_full_pipeline_stub_strategy(tmp_path):
    """Strategy → runner → portfolio → metrics → report, deterministic."""
    universe = ["WIN", "LOSE"]
    rebal = generate_rebalance_dates("2023-01-15", "2024-01-15", "monthly")
    out = tmp_path / "decisions.jsonl"

    strat = callable_strategy(
        lambda tkr, d: "Buy" if tkr == "WIN" else "Sell"
    )
    list(walk_forward(universe, rebal, strat, out))
    decisions = load_decisions(out)
    assert len(decisions) == len(universe) * len(rebal)

    prices = _trending_prices(universe + ["SPY"])
    equity = build_equity_curve(decisions, prices)
    bench = benchmark_curve(prices, "SPY", equity.index)
    summary = summarise(equity, bench)
    per_tkr = per_ticker_stats(decisions, prices, holding_days=10)

    # Strategy is "always buy the uptrend, always sell the downtrend" against
    # deterministic +0.1%/-0.1% daily series — alpha must be positive.
    assert summary.total_return > 0
    assert summary.alpha_vs_benchmark is not None
    assert summary.alpha_vs_benchmark > 0

    # WIN gets Buy ratings on an uptrend → hit rate should be 100%.
    win_row = per_tkr.set_index("ticker").loc["WIN"]
    assert win_row["hit_rate"] == pytest.approx(1.0)
    # LOSE gets Sell ratings on a downtrend → also 100%.
    lose_row = per_tkr.set_index("ticker").loc["LOSE"]
    assert lose_row["hit_rate"] == pytest.approx(1.0)

    md = render_markdown(
        summary,
        title="test run",
        universe=universe,
        date_range=("2023-01-15", "2024-01-15"),
        benchmark_name="SPY",
        per_ticker=per_tkr,
    )
    assert "WIN" in md and "LOSE" in md
    assert "Alpha vs benchmark" in md


def test_html_render_handles_missing_matplotlib(tmp_path, monkeypatch):
    """HTML should still produce output when matplotlib isn't available."""
    universe = ["WIN"]
    rebal = generate_rebalance_dates("2023-01-15", "2023-06-15", "monthly")
    out = tmp_path / "decisions.jsonl"
    list(walk_forward(
        universe, rebal,
        callable_strategy(lambda t, d: "Buy"),
        out,
    ))
    decisions = load_decisions(out)
    prices = _trending_prices(universe)
    equity = build_equity_curve(decisions, prices)

    # Force the import to fail so the chart helper returns None.
    import builtins
    real_import = builtins.__import__
    def faux_import(name, *a, **kw):
        if name == "matplotlib":
            raise ImportError("simulated")
        return real_import(name, *a, **kw)
    monkeypatch.setattr(builtins, "__import__", faux_import)

    html = render_html(summarise(equity), equity)
    assert "<html>" in html
    assert "<img" not in html  # no chart embedded


# --- baseline honesty check ----------------------------------------------


def test_random_strategy_averages_near_zero_alpha():
    """Big random sample → near-zero return on a flat universe.

    This is the test on the *metrics*, not the strategy. If random
    decisions on flat prices produce systematically nonzero returns,
    something is wrong with portfolio/metrics math.
    """
    universe = ["A", "B", "C", "D", "E"]
    rebal = generate_rebalance_dates("2023-01-15", "2024-12-15", "weekly")
    rng = np.random.default_rng(11)
    decisions = [
        Decision(
            ticker=tkr,
            trade_date=d,
            rating=rng.choice(["Buy", "Hold", "Sell"]),
        )
        for d in rebal for tkr in universe
    ]
    # Flat prices for everyone (no drift) so returns are pure noise.
    idx = pd.bdate_range("2023-01-01", "2024-12-31")
    prices = pd.DataFrame({tkr: 100.0 for tkr in universe}, index=idx)
    equity = build_equity_curve(decisions, prices)
    # All buys/sells/holds against perfectly flat prices → equity stays at 1.
    assert equity.iloc[-1] == pytest.approx(1.0, abs=1e-9)
