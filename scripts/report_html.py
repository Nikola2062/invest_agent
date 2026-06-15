"""Build a single self-contained HTML report from one or more ticker runs.

A "run entry" is a dict:
    {
        "ticker": str, "date": str, "decision": str,   # short rating
        "stats": {"llm_calls", "tool_calls", "tokens_in", "tokens_out",
                  "tokens_total", "est_cost_usd", "elapsed_seconds"},
        "state": <final_state dict from propagate()>,
    }

The report adapts: one ticker -> single-section report; N tickers -> tabbed.
No external/runtime dependencies; Markdown is rendered to HTML in-process so
the file works fully offline.
"""
from __future__ import annotations

import html
import re
from datetime import datetime, timezone


# --------------------------- Markdown -> HTML ---------------------------------

def _inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"`([^`]+?)`", r"<code>\1</code>", text)
    # italics: single * or _ not part of ** ; keep conservative
    text = re.sub(r"(?<![\*\w])\*(?!\*)([^\*\n]+?)\*(?!\*)", r"<em>\1</em>", text)
    return text


def _is_table_sep(line: str) -> bool:
    cells = [c.strip() for c in line.strip().strip("|").split("|")]
    return len(cells) >= 1 and all(re.fullmatch(r":?-{2,}:?", c or "-") for c in cells if c != "") and "-" in line


def _row_cells(line: str):
    return [c.strip() for c in line.strip().strip("|").split("|")]


def md_to_html(md: str) -> str:
    if not md:
        return "<p class='muted'>(empty)</p>"
    lines = str(md).replace("\r\n", "\n").split("\n")
    out, i, n = [], 0, len(lines)
    para: list[str] = []

    def flush_para():
        if para:
            out.append("<p>" + "<br>".join(_inline(x) for x in para) + "</p>")
            para.clear()

    while i < n:
        line = lines[i]
        stripped = line.strip()

        # table: header + separator on next line
        if "|" in line and i + 1 < n and _is_table_sep(lines[i + 1]):
            flush_para()
            header = _row_cells(line)
            i += 2
            rows = []
            while i < n and "|" in lines[i] and lines[i].strip():
                rows.append(_row_cells(lines[i]))
                i += 1
            thead = "".join(f"<th>{_inline(c)}</th>" for c in header)
            body = ""
            for r in rows:
                cells = (r + [""] * len(header))[: len(header)]
                body += "<tr>" + "".join(f"<td>{_inline(c)}</td>" for c in cells) + "</tr>"
            out.append(f"<table><thead><tr>{thead}</tr></thead><tbody>{body}</tbody></table>")
            continue

        if not stripped:
            flush_para()
            i += 1
            continue

        if re.fullmatch(r"(-{3,}|\*{3,}|_{3,})", stripped):
            flush_para()
            out.append("<hr>")
            i += 1
            continue

        m = re.match(r"(#{1,6})\s+(.*)", stripped)
        if m:
            flush_para()
            lvl = len(m.group(1))
            out.append(f"<h{lvl}>{_inline(m.group(2))}</h{lvl}>")
            i += 1
            continue

        if re.match(r"[-*+]\s+", stripped) or re.match(r"\d+[.)]\s+", stripped):
            flush_para()
            ordered = bool(re.match(r"\d+[.)]\s+", stripped))
            items = []
            while i < n and lines[i].strip() and (
                re.match(r"[-*+]\s+", lines[i].strip()) or re.match(r"\d+[.)]\s+", lines[i].strip())
            ):
                item = re.sub(r"^([-*+]|\d+[.)])\s+", "", lines[i].strip())
                items.append(f"<li>{_inline(item)}</li>")
                i += 1
            tag = "ol" if ordered else "ul"
            out.append(f"<{tag}>{''.join(items)}</{tag}>")
            continue

        para.append(stripped)
        i += 1

    flush_para()
    return "\n".join(out)


# ------------------------------ Report build ----------------------------------

def _badge_class(decision: str) -> str:
    d = (decision or "").lower()
    if any(w in d for w in ("overweight", "buy", "bullish")):
        return "pos"
    if any(w in d for w in ("underweight", "sell", "bearish")):
        return "neg"
    return "neu"


def _get(state, *keys, default=""):
    cur = state
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur if isinstance(cur, str) else default


def _card(title, md, open_=False, sub=""):
    badge = f"<span class='sub'>{html.escape(sub)}</span>" if sub else ""
    return (
        f"<details class='card'{' open' if open_ else ''}>"
        f"<summary><span class='card-title'>{html.escape(title)}</span>{badge}</summary>"
        f"<div class='card-body'>{md_to_html(md)}</div></details>"
    )


def _risk_class(level: str) -> str:
    l = (level or "").upper()
    if l in ("HIGH", "CRITICAL"):
        return "neg"
    if l == "MEDIUM":
        return "neu"
    return "pos"


# Each indicator's (horizon, group). Horizon = how timely the gauge is; group =
# which indicators measure the same thing (so the tally is not read as N
# independent confirmations). Metadata lives here, not in the snapshot data model.
_MACRO_META = {
    "Shiller CAPE":      ("Strategic", "Valuation"),
    "Buffett Indicator": ("Strategic", "Valuation"),
    "Cboe VIX":          ("Tactical",  "Volatility / Sentiment"),
    "CNN Fear & Greed":  ("Tactical",  "Volatility / Sentiment"),
    "Margin Debt YoY":   ("Strategic", "Volatility / Sentiment"),
    "10Y-2Y Spread":     ("Tactical",  "Credit / Rates"),
    "HY OAS":            ("Macro",     "Credit / Rates"),
    "PMI + Sahm Rule":   ("Macro",     "Growth"),
}
_MACRO_GROUP_ORDER = ["Valuation", "Volatility / Sentiment", "Credit / Rates", "Growth"]
_HORIZON_NOTE = {
    "Tactical": "days–weeks",
    "Strategic": "years",
    "Macro": "monthly, lagged",
}


def _macro_section(macro) -> str:
    """Render the market-wide macro / crash-risk snapshot as a top-of-report card.

    ``macro`` is the dict from ``macro_snapshot.build_snapshot`` (or None to skip).
    Indicators are grouped by what they measure (correlated gauges sit together so
    the tally is not over-read) and tagged by horizon (tactical vs strategic), since
    valuation gauges have ~no power over the days-to-weeks horizon a report operates on.
    """
    if not macro:
        return ""
    inds = macro.get("indicators") or []

    def _row(ind):
        horizon, _ = _MACRO_META.get(ind["name"], ("", ""))
        hnote = _HORIZON_NOTE.get(horizon, "")
        htext = f"{horizon} ({hnote})" if horizon else "—"
        return (
            "<tr>"
            f"<td>{ind['light']['emoji']} {html.escape(ind['name'])}</td>"
            f"<td><strong>{html.escape(ind.get('display', ''))}</strong></td>"
            f"<td class='muted'>{html.escape(htext)}</td>"
            f"<td class='muted'>{html.escape(ind.get('threshold', ''))}</td>"
            f"<td>{html.escape(ind['light']['label'])}</td>"
            "</tr>"
        )

    # Render grouped (known groups in order, then any indicator with no metadata).
    by_group = {g: [] for g in _MACRO_GROUP_ORDER}
    ungrouped = []
    for ind in inds:
        _, group = _MACRO_META.get(ind["name"], ("", ""))
        (by_group[group] if group in by_group else ungrouped).append(ind)
    parts = []
    for group in _MACRO_GROUP_ORDER:
        members = by_group[group]
        if not members:
            continue
        parts.append(f"<tr class='macro-group'><td colspan='5'>{html.escape(group)}</td></tr>")
        parts.extend(_row(ind) for ind in members)
    parts.extend(_row(ind) for ind in ungrouped)
    rows = "".join(parts)
    tally = macro.get("tally") or {}
    sp = macro.get("sp500") or {}
    level = macro.get("risk_level", "N/A")
    sp_line = "N/A"
    if sp.get("drawdown_pct") is not None:
        sp_line = (
            f"{sp['current']:.2f} "
            f"(ATH {sp['ath']:.2f}, {sp['drawdown_pct']:+.2f}%)"
        )
    body = f"""
      <div class="macro-head">
        <span class="pill {_risk_class(level)}">Risk: {html.escape(str(level))}</span>
        <span class="macro-tally">🔴 {tally.get('red', 0)} · 🟡 {tally.get('yellow', 0)} · 🟢 {tally.get('green', 0)}</span>
        <span class="muted">as of {html.escape(macro.get('timestamp', ''))}</span>
      </div>
      <p class="muted">Strategic backdrop, not a trading trigger — valuation gauges
      (CAPE, Buffett) signal multi-year expected return, not days-to-weeks timing.</p>
      <table class="macro-table">
        <thead><tr><th>Indicator</th><th>Value</th><th>Horizon</th><th>Threshold</th><th>Light</th></tr></thead>
        <tbody>{rows}</tbody>
      </table>
      <p><strong>S&amp;P 500:</strong> {html.escape(sp_line)}</p>
      <p><strong>Action:</strong> {html.escape(macro.get('action', ''))}</p>
      <p class="muted">US market context — shared across all tickers in this batch. The
      red/yellow/green tally is <em>indicative, not a weighted model</em>: indicators within
      a group are correlated (CAPE ≈ Buffett; VIX ≈ Fear &amp; Greed), so several "reds" may
      be one signal double-counted, not independent confirmations.</p>"""
    return (
        "<details class='card macro-card' open>"
        "<summary><span class='card-title'>🌐 Market Risk Context</span>"
        f"<span class='sub'>{html.escape(str(level))} risk</span></summary>"
        f"<div class='card-body'>{body}</div></details>"
    )


def _ticker_section(entry, idx, active):
    state = entry.get("state") or {}
    s = entry.get("stats") or {}
    ticker = entry["ticker"]
    decision = entry.get("decision", "")
    bcls = _badge_class(decision)

    metrics = f"""
      <div class="metrics">
        <div class="metric"><span class="m-val">{decision or '—'}</span><span class="m-lab">Rating</span></div>
        <div class="metric"><span class="m-val">${s.get('est_cost_usd', 0):.4f}</span><span class="m-lab">Est. cost</span></div>
        <div class="metric"><span class="m-val">{s.get('tokens_total', 0):,}</span><span class="m-lab">Tokens</span></div>
        <div class="metric"><span class="m-val">{s.get('llm_calls', 0)}</span><span class="m-lab">LLM calls</span></div>
        <div class="metric"><span class="m-val">{s.get('elapsed_seconds', 0):.0f}s</span><span class="m-lab">Elapsed</span></div>
      </div>"""

    cards = [
        _card("📉 Final Trade Decision (Portfolio Manager)", _get(state, "final_trade_decision"), open_=True),
        _card("🧭 Research Manager Verdict", _get(state, "investment_debate_state", "judge_decision")),
        _card("💼 Trader Plan", _get(state, "trader_investment_decision")),
        _card("📈 Market / Technical Analyst", _get(state, "market_report")),
        _card("💬 Sentiment Analyst", _get(state, "sentiment_report")),
        _card("📰 News Analyst", _get(state, "news_report")),
        _card("📊 Fundamentals Analyst", _get(state, "fundamentals_report")),
        _card("🐂 Bull Researcher (transcript)", _get(state, "investment_debate_state", "bull_history")),
        _card("🐻 Bear Researcher (transcript)", _get(state, "investment_debate_state", "bear_history")),
        _card("⚖️ Risk Debate (transcript)", _get(state, "risk_debate_state", "history")),
    ]

    return f"""
    <section class="panel{' active' if active else ''}" id="panel-{idx}">
      <div class="panel-head">
        <h2>{html.escape(ticker)} <span class="pill {bcls}">{html.escape(decision or 'N/A')}</span></h2>
        <div class="dateline">Analysis date: {html.escape(entry.get('date',''))}</div>
      </div>
      {metrics}
      {''.join(cards)}
    </section>"""


def build_report(runs, batch_date="", pricing=None, macro=None) -> str:
    runs = [r for r in runs if r.get("state")]
    pricing = pricing or {}
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    total_cost = sum((r.get("stats") or {}).get("est_cost_usd", 0) for r in runs)
    total_tok = sum((r.get("stats") or {}).get("tokens_total", 0) for r in runs)
    tickers = [r["ticker"] for r in runs]
    multi = len(runs) > 1

    # summary cards
    summary_cards = "".join(
        f"""<button class="sumcard {_badge_class(r.get('decision',''))}" data-target="{i}">
              <span class="sc-ticker">{html.escape(r['ticker'])}</span>
              <span class="sc-rating">{html.escape(r.get('decision','—'))}</span>
              <span class="sc-cost">${(r.get('stats') or {}).get('est_cost_usd',0):.4f} · {(r.get('stats') or {}).get('tokens_total',0):,} tok</span>
            </button>""" for i, r in enumerate(runs)
    )

    tabs = ""
    if multi:
        tabs = "<nav class='tabs'>" + "".join(
            f"<button class='tab{' active' if i == 0 else ''}' data-target='{i}'>"
            f"{html.escape(r['ticker'])}</button>" for i, r in enumerate(runs)
        ) + "</nav>"

    panels = "".join(_ticker_section(r, i, active=(i == 0)) for i, r in enumerate(runs))

    title = f"TradingAgents Report — {', '.join(tickers)}"

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title>
<style>
:root {{
  --bg:#0f1115; --panel:#171a21; --card:#1d212b; --line:#2a2f3a;
  --txt:#e6e8ec; --muted:#9aa3b2; --accent:#5b8cff;
  --pos:#1f9d57; --pos-bg:#11301f; --neg:#e0533d; --neg-bg:#33140f;
  --neu:#d9a128; --neu-bg:#322611;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--txt);
  font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }}
header.top {{ padding:28px 32px 20px; border-bottom:1px solid var(--line);
  background:linear-gradient(180deg,#141823,#0f1115); }}
header.top h1 {{ margin:0 0 4px; font-size:22px; letter-spacing:.2px; }}
header.top .meta {{ color:var(--muted); font-size:13px; }}
.wrap {{ max-width:1040px; margin:0 auto; padding:24px 32px 64px; }}
.totals {{ display:flex; gap:24px; flex-wrap:wrap; margin:18px 0 26px; }}
.totals .t {{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:14px 18px; min-width:150px; }}
.totals .t b {{ display:block; font-size:20px; }}
.totals .t span {{ color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.5px; }}
.sumgrid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); gap:14px; margin-bottom:30px; }}
.sumcard {{ text-align:left; cursor:pointer; background:var(--card); border:1px solid var(--line);
  border-left:4px solid var(--neu); border-radius:12px; padding:14px 16px; color:var(--txt); transition:.15s; }}
.sumcard:hover {{ border-color:var(--accent); transform:translateY(-1px); }}
.sumcard.pos {{ border-left-color:var(--pos); }} .sumcard.neg {{ border-left-color:var(--neg); }}
.sumcard.neu {{ border-left-color:var(--neu); }}
.sc-ticker {{ display:block; font-size:18px; font-weight:700; }}
.sc-rating {{ display:block; font-weight:600; margin:2px 0; }}
.sc-cost {{ display:block; color:var(--muted); font-size:12px; }}
.sumcard.pos .sc-rating {{ color:#5fd99a; }} .sumcard.neg .sc-rating {{ color:#ff8a73; }}
.sumcard.neu .sc-rating {{ color:#f0c560; }}
.tabs {{ display:flex; gap:6px; flex-wrap:wrap; border-bottom:1px solid var(--line); margin-bottom:22px; }}
.tab {{ background:none; border:none; color:var(--muted); padding:10px 16px; cursor:pointer;
  font-size:15px; border-bottom:2px solid transparent; }}
.tab:hover {{ color:var(--txt); }}
.tab.active {{ color:var(--txt); border-bottom-color:var(--accent); font-weight:600; }}
.panel {{ display:none; }} .panel.active {{ display:block; }}
.panel-head h2 {{ margin:0 0 2px; font-size:24px; }}
.dateline {{ color:var(--muted); font-size:13px; margin-bottom:16px; }}
.pill {{ font-size:13px; font-weight:600; padding:3px 12px; border-radius:999px; vertical-align:middle; margin-left:8px; }}
.pill.pos {{ background:var(--pos-bg); color:#5fd99a; }} .pill.neg {{ background:var(--neg-bg); color:#ff8a73; }}
.pill.neu {{ background:var(--neu-bg); color:#f0c560; }}
.metrics {{ display:flex; gap:12px; flex-wrap:wrap; margin:0 0 22px; }}
.metric {{ background:var(--card); border:1px solid var(--line); border-radius:10px; padding:10px 16px; min-width:96px; }}
.metric .m-val {{ display:block; font-size:18px; font-weight:600; }}
.metric .m-lab {{ display:block; color:var(--muted); font-size:11px; text-transform:uppercase; letter-spacing:.5px; }}
details.card {{ background:var(--card); border:1px solid var(--line); border-radius:12px; margin-bottom:12px; overflow:hidden; }}
details.card > summary {{ cursor:pointer; list-style:none; padding:14px 18px; display:flex; align-items:center;
  justify-content:space-between; font-weight:600; user-select:none; }}
details.card > summary::-webkit-details-marker {{ display:none; }}
details.card > summary:hover {{ background:#222632; }}
.card-title {{ font-size:15px; }}
details.card .sub {{ color:var(--muted); font-weight:400; font-size:12px; }}
.card-body {{ padding:4px 22px 18px; border-top:1px solid var(--line); }}
.card-body h1,.card-body h2,.card-body h3,.card-body h4 {{ line-height:1.3; margin:18px 0 8px; }}
.card-body h1 {{ font-size:20px; }} .card-body h2 {{ font-size:18px; }} .card-body h3 {{ font-size:16px; }}
.card-body table {{ border-collapse:collapse; width:100%; margin:12px 0; font-size:13.5px; }}
.card-body th,.card-body td {{ border:1px solid var(--line); padding:7px 10px; text-align:left; vertical-align:top; }}
.card-body thead th {{ background:#222736; }}
.card-body tbody tr:nth-child(even) {{ background:#1a1e27; }}
.card-body code {{ background:#0c0e13; padding:1px 6px; border-radius:5px; font-size:13px; }}
.card-body hr {{ border:none; border-top:1px solid var(--line); margin:16px 0; }}
.card-body ul,.card-body ol {{ padding-left:22px; }}
.macro-card {{ margin-bottom:26px; border-color:var(--accent); }}
.macro-head {{ display:flex; gap:14px; align-items:center; flex-wrap:wrap; margin:6px 0 14px; }}
.macro-tally {{ font-size:14px; font-weight:600; }}
.macro-table {{ border-collapse:collapse; width:100%; margin:6px 0 4px; font-size:13.5px; }}
.macro-table th,.macro-table td {{ border:1px solid var(--line); padding:7px 10px; text-align:left; }}
.macro-table thead th {{ background:#222736; }}
.macro-table tbody tr:nth-child(even) {{ background:#1a1e27; }}
.macro-table tr.macro-group td {{ background:#222736; font-weight:600; color:var(--muted); letter-spacing:.04em; text-transform:uppercase; font-size:11.5px; }}
.muted {{ color:var(--muted); }}
footer {{ color:var(--muted); font-size:12px; text-align:center; padding:24px; border-top:1px solid var(--line); }}
</style>
</head>
<body>
<header class="top">
  <h1>📡 TradingAgents Analysis Report</h1>
  <div class="meta">{html.escape(', '.join(tickers))} · batch date {html.escape(batch_date)} · generated {generated}</div>
</header>
<div class="wrap">
  <div class="totals">
    <div class="t"><b>{len(runs)}</b><span>Tickers</span></div>
    <div class="t"><b>${total_cost:.4f}</b><span>Total est. cost</span></div>
    <div class="t"><b>{total_tok:,}</b><span>Total tokens</span></div>
    <div class="t"><b>{html.escape(str(pricing.get('input_per_1m','?')))}/{html.escape(str(pricing.get('output_per_1m','?')))}</b><span>$/1M in·out</span></div>
  </div>
  {_macro_section(macro)}
  <div class="sumgrid">{summary_cards}</div>
  {tabs}
  {panels}
</div>
<footer>Generated by TradingAgents · for research only — not financial advice.</footer>
<script>
  function activate(idx) {{
    document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
    document.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
    var panel = document.getElementById('panel-' + idx);
    if (panel) panel.classList.add('active');
    var tab = document.querySelector('.tab[data-target="' + idx + '"]');
    if (tab) tab.classList.add('active');
    window.scrollTo({{top:0, behavior:'smooth'}});
  }}
  document.querySelectorAll('.tab,[data-target]').forEach(el =>
    el.addEventListener('click', () => activate(el.dataset.target)));
</script>
</body>
</html>"""
