"""Tearsheet rendering for a completed backtest.

Markdown is the canonical format — readable in any terminal, copy-pastes
into PRs, and works without matplotlib. HTML output is a thin wrapper that
embeds the same content plus an optional inline equity-curve PNG (only if
matplotlib is importable; we don't add it as a hard dependency).

Per-ticker hit rate breaks down which names the strategy actually got
right — when total alpha is mediocre, this is usually where the signal
hides (or doesn't).
"""

from __future__ import annotations

import base64
import io
from collections import defaultdict
from typing import Optional, Sequence

import pandas as pd

from .metrics import Summary
from .strategy import Decision


# --- per-ticker breakdown -------------------------------------------------


def per_ticker_stats(
    decisions: Sequence[Decision],
    prices: pd.DataFrame,
    holding_days: int = 21,
) -> pd.DataFrame:
    """Compute hit rate and mean forward return for each (ticker, rating) bucket.

    For each decision, forward return is measured from the rebalance day's
    close to the close ``holding_days`` trading days later. A trade "hits"
    if Buy/Overweight ratings see positive forward returns or
    Sell/Underweight see negative ones.

    Hold ratings are excluded — there's no signal to score.
    """
    if not decisions or prices.empty:
        return pd.DataFrame()

    rows = []
    for d in decisions:
        if d.rating == "Hold":
            continue
        if d.ticker not in prices.columns:
            continue
        series = prices[d.ticker].dropna()
        ts = pd.Timestamp(d.trade_date)
        future = series.loc[series.index >= ts]
        if len(future) < holding_days + 1:
            continue
        p_start = future.iloc[0]
        p_end = future.iloc[holding_days]
        fwd_ret = float(p_end / p_start - 1)
        bullish = d.rating in ("Buy", "Overweight")
        hit = (bullish and fwd_ret > 0) or (not bullish and fwd_ret < 0)
        rows.append({
            "ticker": d.ticker,
            "rating": d.rating,
            "fwd_return": fwd_ret,
            "hit": hit,
        })

    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    grouped = df.groupby("ticker").agg(
        n_trades=("hit", "size"),
        hit_rate=("hit", "mean"),
        mean_fwd_return=("fwd_return", "mean"),
    ).reset_index()
    return grouped.sort_values("n_trades", ascending=False)


# --- markdown rendering ---------------------------------------------------


def _fmt_pct(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x * 100:+.2f}%"


def _fmt_ratio(x: float | None) -> str:
    if x is None or pd.isna(x):
        return "—"
    return f"{x:.2f}"


def render_markdown(
    summary: Summary,
    *,
    title: str = "Backtest Tearsheet",
    universe: Sequence[str] | None = None,
    date_range: tuple[str, str] | None = None,
    benchmark_name: str | None = None,
    per_ticker: pd.DataFrame | None = None,
) -> str:
    """Render a Markdown tearsheet from a metrics ``Summary``."""
    lines = [f"# {title}", ""]

    if date_range:
        lines.append(f"**Period:** {date_range[0]} → {date_range[1]}")
    if universe:
        n = len(universe)
        head = ", ".join(universe[:8])
        suffix = f" (+{n - 8} more)" if n > 8 else ""
        lines.append(f"**Universe ({n}):** {head}{suffix}")
    if benchmark_name:
        lines.append(f"**Benchmark:** {benchmark_name}")
    lines.append("")

    lines += [
        "## Performance",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Periods | {summary.n_periods} |",
        f"| Total return | {_fmt_pct(summary.total_return)} |",
        f"| CAGR | {_fmt_pct(summary.cagr)} |",
        f"| Volatility (ann.) | {_fmt_pct(summary.volatility)} |",
        f"| Sharpe | {_fmt_ratio(summary.sharpe)} |",
        f"| Sortino | {_fmt_ratio(summary.sortino)} |",
        f"| Max drawdown | {_fmt_pct(summary.max_drawdown)} |",
        f"| Calmar | {_fmt_ratio(summary.calmar)} |",
        f"| Hit rate (periods +) | {_fmt_pct(summary.hit_rate)} |",
    ]
    if summary.benchmark_total_return is not None:
        lines += [
            f"| Benchmark total return | {_fmt_pct(summary.benchmark_total_return)} |",
            f"| Alpha vs benchmark | {_fmt_pct(summary.alpha_vs_benchmark)} |",
            f"| Information ratio | {_fmt_ratio(summary.information_ratio)} |",
        ]
    lines.append("")

    if per_ticker is not None and not per_ticker.empty:
        lines += ["## Per-ticker breakdown", "",
                  "| Ticker | Trades | Hit rate | Mean fwd return |",
                  "|---|---:|---:|---:|"]
        for _, row in per_ticker.iterrows():
            lines.append(
                f"| {row['ticker']} | {int(row['n_trades'])} | "
                f"{_fmt_pct(row['hit_rate'])} | {_fmt_pct(row['mean_fwd_return'])} |"
            )
        lines.append("")

    return "\n".join(lines)


# --- HTML rendering -------------------------------------------------------


def _equity_chart_png(equity: pd.Series, benchmark: pd.Series | None) -> bytes | None:
    """Render a small equity-curve PNG. Returns None if matplotlib is absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(equity.index, equity.values, label="Strategy", linewidth=2)
    if benchmark is not None and not benchmark.empty:
        ax.plot(benchmark.index, benchmark.values, label="Benchmark",
                linewidth=1.5, linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Equity (normalised)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return buf.getvalue()


def render_html(
    summary: Summary,
    equity: pd.Series,
    benchmark: pd.Series | None = None,
    **markdown_kwargs,
) -> str:
    """Render an HTML tearsheet, embedding the equity chart inline if possible."""
    md = render_markdown(summary, **markdown_kwargs)
    png = _equity_chart_png(equity, benchmark)
    chart_html = ""
    if png is not None:
        b64 = base64.b64encode(png).decode("ascii")
        chart_html = (
            f'<img src="data:image/png;base64,{b64}" '
            f'alt="Equity curve" style="max-width:100%;"/>'
        )

    # Minimal HTML; user can wrap with their own template if they want.
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<style>body{font-family:system-ui,sans-serif;max-width:900px;"
        "margin:2em auto;padding:0 1em;line-height:1.5;}"
        "table{border-collapse:collapse;}th,td{padding:4px 12px;"
        "border-bottom:1px solid #eee;}th{text-align:left;background:#f6f6f6;}"
        "</style></head><body>"
        f"{chart_html}"
        f"<pre style='white-space:pre-wrap;font-family:inherit;'>{md}</pre>"
        "</body></html>"
    )
