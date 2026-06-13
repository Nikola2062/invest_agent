"""Monitor agent — thesis-break detector for held positions.

Pure deterministic (no LLM). Reads the audit trail. Compares the current run
to the most recent prior run for each held position. Detects step-change events
that suggest the thesis has materially shifted, and writes ThesisBreakAlert rows.

This runs after analyze_all() in the scheduler. It also exposes a CLI for
on-demand checks.

Triggers (any one fires an alert):
  - quality_drop:        quality_score fell by >= 2.0 points
  - mos_flip:            margin_of_safety flipped from >= +10% to <= -5%
  - drawdown_jump:       P(dd>=20%) rose by >= 0.20 absolute
  - tactical_escalation: tactical level escalated by >= 2 steps
  - devil_veto:          DA verdict flipped to "veto"
  - sentiment_drop:      sentiment_score fell by >= 0.5
  - vol_spike:           realized_vol relative jump >= 50%
  - rebuy_stale:         trim/exit > N days ago, rebuy band never reached → re-evaluate
  - technical_signal_flip: Technical Division composite flipped polarity (bullish↔bearish)

Severity:
  - critical: devil_veto, mos_flip, tactical escalation to BLACK_EXIT
  - warning:  quality_drop, drawdown_jump, tactical escalation to RED_DEFENSIVE, rebuy_stale
  - info:     sentiment_drop, vol_spike, smaller tactical escalations
"""
from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from src.config.loader import load_portfolio, load_risk_policy
from src.storage.audit import get_recent_runs, record_alert

log = logging.getLogger(__name__)


@dataclass
class ThesisBreak:
    symbol: str
    severity: str           # critical / warning / info
    category: str           # one of the triggers listed above
    summary: str
    evidence: dict
    alert_id: Optional[int] = None


# Tactical level ordering for escalation comparison
_LEVEL_ORDER = {
    None: 0,
    "YELLOW_WATCH": 1,
    "ORANGE_TRIM": 2,
    "RED_DEFENSIVE": 3,
    "BLACK_EXIT": 4,
}


def _row_get(row: sqlite3.Row, key: str):
    try:
        return row[key]
    except (IndexError, KeyError):
        return None


def _detect_for_symbol(curr: sqlite3.Row, prev: sqlite3.Row) -> list[ThesisBreak]:
    """Compare two audit rows for one symbol and emit any thesis-break alerts."""
    symbol = curr["symbol"]
    breaks: list[ThesisBreak] = []

    # 1. quality_drop
    cq = _row_get(curr, "quality_score")
    pq = _row_get(prev, "quality_score")
    if cq is not None and pq is not None and (pq - cq) >= 2.0:
        breaks.append(ThesisBreak(
            symbol=symbol,
            severity="warning",
            category="quality_drop",
            summary=f"Quality score fell {pq:.1f} → {cq:.1f} ({pq - cq:+.1f}) — fundamental thesis may be breaking",
            evidence={"prev_quality": pq, "curr_quality": cq, "prev_timestamp": prev["timestamp_utc"]},
        ))

    # 2. mos_flip
    cm = _row_get(curr, "margin_of_safety_pct")
    pm = _row_get(prev, "margin_of_safety_pct")
    if cm is not None and pm is not None and pm >= 10.0 and cm <= -5.0:
        breaks.append(ThesisBreak(
            symbol=symbol,
            severity="critical",
            category="mos_flip",
            summary=f"Margin of safety flipped {pm:+.1f}% → {cm:+.1f}% — no longer undervalued",
            evidence={"prev_mos": pm, "curr_mos": cm},
        ))

    # 3. drawdown_jump
    cd = _row_get(curr, "p_dd_20")
    pd_prev = _row_get(prev, "p_dd_20")
    if cd is not None and pd_prev is not None and (cd - pd_prev) >= 0.20:
        breaks.append(ThesisBreak(
            symbol=symbol,
            severity="warning",
            category="drawdown_jump",
            summary=f"P(drawdown ≥ 20%) jumped {pd_prev:.0%} → {cd:.0%} (+{(cd - pd_prev) * 100:.0f}pp)",
            evidence={"prev_p_dd_20": pd_prev, "curr_p_dd_20": cd},
        ))

    # 4. tactical_escalation
    cl = _row_get(curr, "tactical_label")
    pl = _row_get(prev, "tactical_label")
    cl_idx = _LEVEL_ORDER.get(cl, 0)
    pl_idx = _LEVEL_ORDER.get(pl, 0)
    if cl_idx - pl_idx >= 2:
        sev = "critical" if cl == "BLACK_EXIT" else ("warning" if cl == "RED_DEFENSIVE" else "info")
        breaks.append(ThesisBreak(
            symbol=symbol,
            severity=sev,
            category="tactical_escalation",
            summary=f"Tactical level escalated {pl or 'NO_ACTION'} → {cl} ({cl_idx - pl_idx} steps)",
            evidence={"prev_level": pl, "curr_level": cl},
        ))

    # 5. devil_veto
    cv = _row_get(curr, "devil_verdict")
    pv = _row_get(prev, "devil_verdict")
    if cv == "veto" and pv != "veto":
        breaks.append(ThesisBreak(
            symbol=symbol,
            severity="critical",
            category="devil_veto",
            summary="Devil's Advocate verdict turned VETO — recommendation should not be acted on",
            evidence={"prev_verdict": pv, "curr_verdict": cv},
        ))

    # 6. sentiment_drop
    cs = _row_get(curr, "sentiment_score")
    ps = _row_get(prev, "sentiment_score")
    if cs is not None and ps is not None and (ps - cs) >= 0.5:
        breaks.append(ThesisBreak(
            symbol=symbol,
            severity="info",
            category="sentiment_drop",
            summary=f"News sentiment dropped {ps:+.2f} → {cs:+.2f}",
            evidence={"prev_sentiment": ps, "curr_sentiment": cs},
        ))

    # 7. vol_spike
    cv_ = _row_get(curr, "realized_vol_pct")
    pv_ = _row_get(prev, "realized_vol_pct")
    if cv_ is not None and pv_ is not None and pv_ > 0 and (cv_ / pv_) - 1.0 >= 0.5:
        breaks.append(ThesisBreak(
            symbol=symbol,
            severity="info",
            category="vol_spike",
            summary=f"Realized vol spiked {pv_:.1f}% → {cv_:.1f}% (+{(cv_/pv_ - 1)*100:.0f}%)",
            evidence={"prev_vol": pv_, "curr_vol": cv_},
        ))

    # 8. technical_signal_flip — composite flipped polarity (bullish ↔ bearish)
    ct = _row_get(curr, "composite_tech_signal")
    pt = _row_get(prev, "composite_tech_signal")
    _BULL = {"bullish", "strong_bullish"}
    _BEAR = {"bearish", "strong_bearish"}
    if (
        ct is not None and pt is not None
        and ((pt in _BULL and ct in _BEAR) or (pt in _BEAR and ct in _BULL))
    ):
        severity = "warning" if "strong" in ct or "strong" in pt else "info"
        breaks.append(ThesisBreak(
            symbol=symbol, severity=severity,
            category="technical_signal_flip",
            summary=f"Technical composite flipped {pt} → {ct} — price-action thesis has changed sides",
            evidence={"prev_signal": pt, "curr_signal": ct},
        ))

    return breaks


_TRIM_OR_EXIT_LEVELS = {"ORANGE_TRIM", "RED_DEFENSIVE", "BLACK_EXIT"}


def _rebuy_band_high_from_row(row: sqlite3.Row) -> Optional[float]:
    """Parse the persisted full_result_json to extract the rebuy_band_high
    that was active at that historical run. Returns None if not present."""
    blob = _row_get(row, "full_result_json")
    if not blob:
        return None
    try:
        data = json.loads(blob)
    except (TypeError, ValueError):
        return None
    tactical = (data.get("if_held") or {}).get("tactical") or {}
    return tactical.get("rebuy_band_high")


def _detect_rebuy_stale(symbol: str, recent: list[sqlite3.Row], staleness_days: int) -> Optional[ThesisBreak]:
    """Find the most recent trim/exit for this symbol. If it was > staleness_days
    ago and the current price still sits above the staged rebuy band, emit a
    rebuy_stale alert. Returns None if no action is needed.
    """
    trim_row: Optional[sqlite3.Row] = None
    for r in recent:
        if _row_get(r, "tactical_label") in _TRIM_OR_EXIT_LEVELS:
            trim_row = r
            break
    if trim_row is None:
        return None

    rebuy_high = _rebuy_band_high_from_row(trim_row)
    if rebuy_high is None:
        return None

    try:
        trim_ts = datetime.fromisoformat(trim_row["timestamp_utc"])
    except (TypeError, ValueError):
        return None

    now = datetime.now(timezone.utc)
    if trim_ts.tzinfo is None:
        trim_ts = trim_ts.replace(tzinfo=timezone.utc)
    elapsed_days = (now - trim_ts).days
    if elapsed_days < staleness_days:
        return None

    current = _row_get(recent[0], "current_price") if recent else None
    if current is None or current <= rebuy_high:
        # Either we have no current price or the band has been reached — no alert.
        return None

    return ThesisBreak(
        symbol=symbol,
        severity="warning",
        category="rebuy_stale",
        summary=(
            f"Rebuy band never reached: {elapsed_days}d since {trim_row['tactical_label']} "
            f"(staged rebuy ≤ {rebuy_high:.2f}, current {current:.2f}). "
            f"Re-evaluate thesis from scratch — don't sit in cash by inertia."
        ),
        evidence={
            "trim_at": trim_row["timestamp_utc"],
            "trim_label": trim_row["tactical_label"],
            "rebuy_band_high": rebuy_high,
            "current_price": current,
            "elapsed_days": elapsed_days,
            "staleness_threshold_days": staleness_days,
        },
    )


def check_holdings() -> list[ThesisBreak]:
    """Scan all held positions for thesis-break events.

    For each holding:
      - Look up the most recent two analysis_runs rows
      - If fewer than 2 runs exist, skip (no prior baseline)
      - Otherwise compare and emit alerts

    Persists each detected break via record_alert(). Returns the list.
    """
    portfolio = load_portfolio()
    policy = load_risk_policy()
    all_breaks: list[ThesisBreak] = []
    for h in portfolio.holdings:
        runs = get_recent_runs(h.symbol, limit=30)  # need depth to find a trim/exit
        breaks: list[ThesisBreak] = []
        if len(runs) >= 2:
            breaks.extend(_detect_for_symbol(runs[0], runs[1]))
        else:
            log.info("Monitor: skipping pair-diff for %s — fewer than 2 audit rows.", h.symbol)

        stale = _detect_rebuy_stale(h.symbol, runs, policy.rebuy_staleness_days)
        if stale is not None:
            breaks.append(stale)

        for b in breaks:
            try:
                aid = record_alert(b.symbol, b.severity, b.category, b.summary, b.evidence)
                b.alert_id = aid
                log.info("Monitor alert: %s [%s] %s", b.symbol, b.severity, b.summary)
            except Exception as e:
                log.warning("Failed to record alert for %s: %s", b.symbol, e)
        all_breaks.extend(breaks)
    return all_breaks
