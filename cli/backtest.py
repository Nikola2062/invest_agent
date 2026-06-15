"""`tradingagents backtest` — run a walk-forward backtest end-to-end.

Defaults are deliberately conservative on cost. The `random` strategy is
free (no LLM calls) and exists as the baseline you compare the `agent`
strategy against. Always start by running `random` to sanity-check the
universe, date range, and plumbing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.markdown import Markdown

console = Console()


def backtest_command(
    universe: str = typer.Option(
        ...,
        "--universe",
        "-u",
        help="Comma-separated tickers, or @path/to/tickers.txt (one per line).",
    ),
    from_date: str = typer.Option(
        ..., "--from", "-f", help="Start date YYYY-MM-DD (inclusive)."
    ),
    to_date: str = typer.Option(
        ..., "--to", "-t", help="End date YYYY-MM-DD (inclusive)."
    ),
    freq: str = typer.Option(
        "monthly",
        "--freq",
        help="Rebalance frequency: daily, weekly, monthly, quarterly.",
    ),
    strategy: str = typer.Option(
        "random",
        "--strategy",
        help="`random` (free baseline) or `agent` (real LLM-driven pipeline).",
    ),
    benchmark: str = typer.Option(
        "SPY", "--benchmark", help="Benchmark ticker for alpha calculation."
    ),
    output_dir: Path = typer.Option(
        Path("backtest_results"),
        "--output",
        "-o",
        help="Directory for decisions JSONL, equity series, and tearsheet.",
    ),
    seed: int = typer.Option(0, "--seed", help="RNG seed for the random strategy."),
    on_error: str = typer.Option(
        "skip", "--on-error", help="`skip` (recommended) or `raise` on strategy failures."
    ),
    html: bool = typer.Option(
        False, "--html", help="Also write tearsheet.html (requires matplotlib for charts)."
    ),
    sizer: str = typer.Option(
        "equal_weight",
        "--sizer",
        help="`equal_weight` (long-only baseline) or `risk_aware` (inv-vol + caps).",
    ),
    target_vol: float = typer.Option(
        0.20, "--target-vol", help="Annualised target vol for risk-aware sizing."
    ),
    max_position: float = typer.Option(
        0.10, "--max-position", help="Per-name |weight| cap (risk-aware)."
    ),
    max_sector: float = typer.Option(
        0.30, "--max-sector", help="Per-sector exposure cap (risk-aware; needs sector data)."
    ),
    max_gross: float = typer.Option(
        1.0, "--max-gross", help="Gross-exposure ceiling (risk-aware)."
    ),
    allow_shorts: bool = typer.Option(
        False, "--allow-shorts", help="If set, Sell/Underweight ratings become short positions."
    ),
):
    """Run a walk-forward backtest and write a tearsheet."""
    from tradingagents.backtest import RandomStrategy
    from tradingagents.backtest.metrics import summarise
    from tradingagents.backtest.portfolio import (
        benchmark_curve,
        build_equity_curve,
        fetch_prices,
    )
    from tradingagents.backtest.report import (
        per_ticker_stats,
        render_html,
        render_markdown,
    )
    from tradingagents.backtest.runner import (
        generate_rebalance_dates,
        load_decisions,
        walk_forward,
    )

    tickers = _resolve_universe(universe)
    dates = generate_rebalance_dates(from_date, to_date, freq)
    console.print(
        f"[bold]Backtest:[/bold] {len(tickers)} tickers × {len(dates)} rebalance dates "
        f"= {len(tickers) * len(dates)} decisions"
    )

    if strategy == "random":
        strat = RandomStrategy(seed=seed)
    elif strategy == "agent":
        from tradingagents.backtest.strategy import AgentStrategy
        strat = AgentStrategy()
    else:
        raise typer.BadParameter(f"unknown strategy {strategy!r}")

    output_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = output_dir / f"decisions_{strategy}.jsonl"
    new_count = 0
    for _decision in walk_forward(
        tickers, dates, strat, decisions_path, on_error=on_error,
    ):
        new_count += 1
        if new_count % 10 == 0:
            console.print(f"  recorded {new_count} new decisions…")

    decisions = load_decisions(decisions_path)
    console.print(f"[green]✓[/green] {len(decisions)} decisions in {decisions_path}")

    # Fetch prices for portfolio construction + benchmark.
    # Pad the end so the final period has next-day prices to mark to.
    pad_end = _pad_date(to_date, days=14)
    prices = fetch_prices(tickers, from_date, pad_end, benchmark=benchmark)

    sizer_fn, sector_map, sizing_cfg = _resolve_sizer(
        sizer,
        tickers,
        target_vol=target_vol,
        max_position=max_position,
        max_sector=max_sector,
        max_gross=max_gross,
        allow_shorts=allow_shorts,
    )

    equity = build_equity_curve(
        decisions, prices,
        sizer=sizer_fn,
        sectors=sector_map,
        sizing_config=sizing_cfg,
    )
    bench = benchmark_curve(prices, benchmark, equity.index) if not equity.empty else None
    summary = summarise(equity, bench)
    per_tkr = per_ticker_stats(decisions, prices)

    md = render_markdown(
        summary,
        title=f"Backtest tearsheet — {strategy} strategy",
        universe=tickers,
        date_range=(from_date, to_date),
        benchmark_name=benchmark,
        per_ticker=per_tkr,
    )
    (output_dir / f"tearsheet_{strategy}.md").write_text(md, encoding="utf-8")
    if not equity.empty:
        equity.to_csv(output_dir / f"equity_{strategy}.csv", header=True)
    console.print()
    console.print(Markdown(md))
    console.print(f"\n[green]✓[/green] Tearsheet: {output_dir / f'tearsheet_{strategy}.md'}")

    if html:
        html_out = render_html(
            summary, equity, bench,
            title=f"Backtest — {strategy}",
            universe=tickers,
            date_range=(from_date, to_date),
            benchmark_name=benchmark,
            per_ticker=per_tkr,
        )
        (output_dir / f"tearsheet_{strategy}.html").write_text(html_out, encoding="utf-8")
        console.print(f"[green]✓[/green] HTML: {output_dir / f'tearsheet_{strategy}.html'}")


def _resolve_sizer(
    name: str,
    tickers: list[str],
    *,
    target_vol: float,
    max_position: float,
    max_sector: float,
    max_gross: float,
    allow_shorts: bool,
):
    """Pick the sizer + pre-load any data it needs.

    Returns ``(sizer_fn, sector_map, sizing_config)`` ready to pass to
    ``build_equity_curve``. For ``equal_weight`` all three are None so
    the legacy path runs untouched.
    """
    if name == "equal_weight":
        return None, None, None
    if name != "risk_aware":
        raise typer.BadParameter(f"unknown sizer {name!r}")

    from tradingagents.portfolio import SizingConfig, risk_aware_sizer
    from tradingagents.portfolio.sectors import fetch_sectors

    cfg = SizingConfig(
        target_vol=target_vol,
        max_position=max_position,
        max_sector_exposure=max_sector,
        max_gross=max_gross,
        allow_shorts=allow_shorts,
    )
    sector_map = fetch_sectors(tickers)
    return risk_aware_sizer, sector_map, cfg


def _resolve_universe(spec: str) -> list[str]:
    """Resolve --universe into a list of tickers.

    Two forms:
      - ``AAPL,MSFT,GOOG`` — inline comma-separated list
      - ``@path/to/file.txt`` — newline-separated file (blank lines / # comments ignored)
    """
    if spec.startswith("@"):
        path = Path(spec[1:]).expanduser()
        if not path.exists():
            raise typer.BadParameter(f"universe file not found: {path}")
        lines = path.read_text(encoding="utf-8").splitlines()
        return [
            ln.strip() for ln in lines
            if ln.strip() and not ln.strip().startswith("#")
        ]
    return [t.strip().upper() for t in spec.split(",") if t.strip()]


def _pad_date(date_str: str, days: int) -> str:
    """Add ``days`` calendar days to a YYYY-MM-DD date string."""
    from datetime import datetime, timedelta
    d = datetime.strptime(date_str, "%Y-%m-%d") + timedelta(days=days)
    return d.strftime("%Y-%m-%d")
