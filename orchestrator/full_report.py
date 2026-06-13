"""Assemble the FULL combined report — the comprehensive document, not the
1-page digest. It stitches together:

  * the full macro gate (every risk_analysis indicator),
  * capital_rotation's own full US + HK reports (embedded verbatim — the 27-section
    documents that project generates), and
  * a full per-stock analysis for every held + watchlist name (stock_report).

Content is English (the source reports + analyst prose are English); the compact
bilingual digest (digest.py) is unchanged.
"""
from __future__ import annotations

import glob
from pathlib import Path

import settings
import stock_report
from adapters import stocks

_CFG = settings.CONFIG


def _macro_section(macro: dict) -> list[str]:
    L = ["## Macro gate"]
    if not macro.get("available"):
        L.append(f"_unavailable: {macro.get('reason')}_")
        return L
    L.append(f"**Risk level:** {macro.get('risk_level')} — {macro.get('action')}")
    tally = macro.get("tally") or {}
    if tally:
        L.append(f"**Tally:** " + ", ".join(f"{k}={v}" for k, v in tally.items()))
    sp = macro.get("sp500") or {}
    if sp:
        L.append(f"**S&P 500:** {sp.get('current')} (ATH {sp.get('ath')}, {sp.get('drawdown_pct')}% from high)")
    inds = macro.get("indicators") or []
    if inds:
        L += ["", "| Indicator | Value | Threshold | Light |", "|---|---|---|---|"]
        for it in inds:
            light = (it.get("light") or {}).get("label", "")
            L.append(f"| {it.get('name')} | {it.get('display', it.get('value'))} "
                     f"| {it.get('threshold', '')} | {light} |")
    return L


def _rotation_sections() -> list[dict]:
    rd = settings.project_dir("rotation") / "reports"
    out: list[dict] = []
    for market, pat in [("US", "*_daily.md"), ("HK", "*_daily_hk.md")]:
        files = sorted(glob.glob(str(rd / pat)))
        if not files:
            continue
        newest = Path(files[-1])
        out.append({"title": f"Capital rotation — {market}",
                    "md": f"## Capital rotation — {market}  _(source: {newest.name})_\n\n"
                          + newest.read_text().strip()})
    if not out:
        out = [{"title": "Capital rotation",
                "md": "## Capital rotation\n_no rotation report on disk yet — run with "
                      "`--schedule` or `--fresh` to generate one._"}]
    return out


def build_sections(data: dict, date_str: str) -> list[dict]:
    """Return the full report as a list of {title, md} sections.

    Used by the PDF renderer for per-section page breaks + a table of contents.
    """
    sections: list[dict] = [{"title": "Macro gate", "md": "\n".join(_macro_section(data["macro"]))}]
    sections += _rotation_sections()

    positions = stocks.read_positions()   # source of truth: stock_analysier/config
    for item in positions["held"] + positions["watchlist"]:
        if not isinstance(item, dict) or "symbol" not in item:
            continue
        sym = item["symbol"]
        title = f"{sym} ({item.get('market', '?')})"
        full = stocks.read_latest_full(sym)
        if full:
            sections.append({"title": title, "md": stock_report.render_full(full)})
        else:
            sections.append({"title": title,
                             "md": f"# {title}\n_No stored analysis yet — request it via the "
                                   f"Telegram bot (`/us {sym}` or `/hk {sym}`), then it appears here._"})
    return sections


def sections_to_markdown(sections: list[dict], date_str: str) -> str:
    """Join sections into one markdown document (for the .md file / translation)."""
    parts = [f"# Investor — FULL report — {date_str}"]
    parts += [s["md"] for s in sections]
    return "\n\n".join(parts)


def build_full(data: dict, date_str: str) -> str:
    """Backward-compatible single-markdown form of the full report."""
    return sections_to_markdown(build_sections(data, date_str), date_str)
