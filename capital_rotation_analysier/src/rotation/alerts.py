"""Alert dispatcher per the project docs §5.6.

8 alert types, file sink as default channel. Telegram is the primary push
channel; it requires TELEGRAM_BOT_TOKEN + TELEGRAM_CHAT_ID env vars and is
skipped without raising when they're absent.

Email/SMTP was removed (2026-06-07) — Gmail SMTP requires an app password
which the user can't generate, and we don't need redundancy beyond Telegram +
the file sink. The schema and dispatcher are simple enough that adding a
Discord/Resend channel later is ~30 lines.

Debounce: an alert of the same (type, headline) within DEBOUNCE_HOURS is suppressed.
Priority levels:
  P1  — fire immediately on any channel (regime flip, strong rotation, validation failure)
  P2  — batch into the daily digest
  P3  — digest-only / heartbeat
"""
from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable

import duckdb

from .config import Config
from .interpret import CONFIDENCE_FLOOR
from .store import connect


DEBOUNCE_HOURS = 12


@dataclass
class Alert:
    alert_type: str
    priority: str        # "P1" | "P2" | "P3"
    headline: str
    body: str
    ts: date
    alert_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


# ============================================================
# Trigger rules
# ============================================================

def _v(signals: dict, name: str, key: str = "score") -> float | None:
    v = signals.get(name, {}).get(key)
    return None if v is None else float(v)


def _conf_ok(signals: dict, name: str) -> bool:
    c = _v(signals, name, "confidence")
    return c is not None and c >= CONFIDENCE_FLOOR


def evaluate_triggers(
    today_signals: dict[str, dict],
    today_regime: dict | None,
    prev_regime: dict | None,
    asof: date,
    verdicts: dict | None = None,
) -> list[Alert]:
    alerts: list[Alert] = []

    # Suppress alerts from signals that failed validation. Regime alert is allowed
    # through because the regime is computed from smoothed multi-signal medians.
    if verdicts:
        today_signals = {
            name: s for name, s in today_signals.items()
            if name not in verdicts or verdicts[name].verdict != "fail"
        }

    # 1. New Macro Regime Detected (P1)
    if today_regime and prev_regime and today_regime["regime"] != prev_regime["regime"]:
        alerts.append(Alert(
            alert_type="regime_change", priority="P1",
            headline=f"Regime flipped: {prev_regime['regime']} → {today_regime['regime']}",
            body=f"Regime confidence {today_regime.get('confidence', 0):.2f}.",
            ts=asof,
        ))

    # 2. Strong Risk-On Rotation (P1)
    roo = _v(today_signals, "risk_on_off")
    if roo is not None and roo > 60 and _conf_ok(today_signals, "risk_on_off"):
        alerts.append(Alert(
            alert_type="strong_risk_on", priority="P1",
            headline=f"Strong risk-on rotation (RoO {roo:+.1f})",
            body="Risk-On/Off score above +60 with confidence ≥0.35.",
            ts=asof,
        ))

    # 3. Strong Risk-Off Rotation (P1)
    if roo is not None and roo < -60 and _conf_ok(today_signals, "risk_on_off"):
        alerts.append(Alert(
            alert_type="strong_risk_off", priority="P1",
            headline=f"Strong risk-off rotation (RoO {roo:+.1f})",
            body="Risk-On/Off score below −60 with confidence ≥0.35.",
            ts=asof,
        ))

    # 4. Inflation Signal Detected (P2)
    inf = _v(today_signals, "inflation")
    if inf is not None and abs(inf) > 50 and _conf_ok(today_signals, "inflation"):
        dirn = "rising" if inf > 0 else "easing"
        alerts.append(Alert(
            alert_type="inflation_signal", priority="P2",
            headline=f"Inflation {dirn} (score {inf:+.1f})",
            body=f"Inflation score crossed ±50 with confidence ≥{CONFIDENCE_FLOOR}.",
            ts=asof,
        ))

    # 5. Growth Acceleration Signal (P2)
    grw = _v(today_signals, "growth")
    if grw is not None and grw > 50 and _conf_ok(today_signals, "growth"):
        alerts.append(Alert(
            alert_type="growth_acceleration", priority="P2",
            headline=f"Growth acceleration (score {grw:+.1f})",
            body="Growth score above +50 with confidence ≥0.35.",
            ts=asof,
        ))

    # 6. Recession Concern Signal (P2 — even though §3.2.7 admits the signal is
    #    slow at daily resolution; we still fire so the user can see it building.)
    rec = _v(today_signals, "recession")
    if rec is not None and rec > 50 and _conf_ok(today_signals, "recession"):
        alerts.append(Alert(
            alert_type="recession_concern", priority="P2",
            headline=f"Recession concern rising (score {rec:+.1f})",
            body="Recession Concern score above +50; remember this signal is slow.",
            ts=asof,
        ))

    # 7. Unusual Volume Event (P2)
    rv = _v(today_signals, "relative_volume")
    if rv is not None and rv > 80 and _conf_ok(today_signals, "relative_volume"):
        alerts.append(Alert(
            alert_type="unusual_volume", priority="P2",
            headline=f"Aggregate volume anomaly (RV {rv:.1f})",
            body="Universe-wide relative-volume score above 80.",
            ts=asof,
        ))

    # 8. Large ETF Flow Event (P1) — placeholder: flows not ingested yet.
    # Will fire when the issuer-direct flow adapter is connected.

    return alerts


def validation_failure_alerts(
    new_verdicts: dict,
    prior_verdicts: dict,
    asof: date,
) -> list[Alert]:
    """Emit a P1 alert whenever a signal newly flips to 'fail', so the operator
    knows to investigate. Pass→Fail or Undetermined→Fail both trigger.
    """
    out: list[Alert] = []
    for name, v in new_verdicts.items():
        if v.verdict != "fail":
            continue
        prior = prior_verdicts.get(name)
        was_failing = prior is not None and prior.verdict == "fail"
        if was_failing:
            continue
        body_lines = [
            f"Signal: {name}",
            f"Reason: {v.reason}",
            f"Forward asset tested: {v.forward_asset or '—'}",
            f"Observations: {v.n_observations}",
        ]
        if v.median_ic_5d is not None:
            body_lines.append(f"Median 5d IC: {v.median_ic_5d:+.3f}")
        if v.pct_windows_pos_ic is not None:
            body_lines.append(f"% windows with IC>0: {v.pct_windows_pos_ic:.0%}")
        if v.hit_rate_overall is not None:
            body_lines.append(f"Overall hit rate: {v.hit_rate_overall:.0%}")
        body_lines.append("")
        body_lines.append("This signal has been suppressed from reports and alerts.")
        body_lines.append("Run `rotate validate` after retraining or with more history to re-test.")
        out.append(Alert(
            alert_type="validation_failure", priority="P1",
            headline=f"Signal failed validation: {name}",
            body="\n".join(body_lines), ts=asof,
        ))
    return out


# ============================================================
# Channels
# ============================================================

def _file_sink(alert: Alert, sink_dir: Path) -> str:
    sink_dir.mkdir(parents=True, exist_ok=True)
    path = sink_dir / f"{alert.ts.isoformat()}_{alert.priority}_{alert.alert_type}_{alert.alert_id}.txt"
    path.write_text(
        f"[{alert.priority}] {alert.headline}\n\n{alert.body}\n\nts={alert.ts.isoformat()}\n",
        encoding="utf-8",
    )
    return f"file://{path}"


# Telegram sendMessage hard limit is 4096 chars; leave headroom for the
# truncation marker so a long P1 body (e.g. validation diagnostics) never 400s.
TELEGRAM_MAX_TEXT = 4096
_TRUNCATION_MARK = "\n… (truncated)"


def _truncate_for_telegram(text: str, limit: int = TELEGRAM_MAX_TEXT) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - len(_TRUNCATION_MARK)] + _TRUNCATION_MARK


def _telegram_send(alert: Alert) -> str | None:
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return None  # silently skip; absence of creds is the explicit opt-out
    text = _truncate_for_telegram(
        f"*[{alert.priority}] {alert.headline}*\n\n{alert.body}\n\n_{alert.ts.isoformat()}_"
    )
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    def _post(payload: dict) -> str:
        data = urllib.parse.urlencode(payload).encode()
        with urllib.request.urlopen(urllib.request.Request(url, data=data), timeout=10) as resp:
            return f"telegram:{resp.status}"

    try:
        return _post({"chat_id": chat_id, "text": text, "parse_mode": "Markdown"})
    except Exception:
        # Unbalanced *_/`* in headline/body (signal names like risk_on_off) make
        # Telegram reject the Markdown parse with HTTP 400. The alert content
        # matters more than the formatting: retry once as plain text.
        try:
            return _post({"chat_id": chat_id, "text": text})
        except Exception as exc:
            return f"telegram_error:{exc}"


def telegram_send_document(file_path: Path, caption: str | None = None) -> str | None:
    """POST a file to Telegram's sendDocument endpoint as multipart/form-data.

    Caller is responsible for absence-of-creds handling — returns None if no
    creds, otherwise returns "telegram_doc:<status>" or an error string.
    """
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID")
    if not token or not chat_id:
        return None
    if not file_path.exists():
        return f"telegram_doc_error:file_not_found:{file_path}"

    boundary = "----rotationbot_" + uuid.uuid4().hex
    body_parts = []

    def _part(name: str, value: str):
        body_parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'
            f"{value}\r\n".encode()
        )

    _part("chat_id", chat_id)
    if caption:
        _part("caption", caption[:1024])  # Telegram limits captions to 1024 chars
        _part("parse_mode", "Markdown")

    # File part
    file_bytes = file_path.read_bytes()
    body_parts.append(
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="document"; filename="{file_path.name}"\r\n'
        f"Content-Type: application/pdf\r\n\r\n".encode()
    )
    body_parts.append(file_bytes)
    body_parts.append(f"\r\n--{boundary}--\r\n".encode())
    body = b"".join(body_parts)

    url = f"https://api.telegram.org/bot{token}/sendDocument"
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            return f"telegram_doc:{resp.status}"
    except Exception as exc:
        return f"telegram_doc_error:{exc}"


# ============================================================
# Dispatcher
# ============================================================

def _is_debounced(con: duckdb.DuckDBPyConnection, alert: Alert) -> bool:
    cutoff = datetime.utcnow() - timedelta(hours=DEBOUNCE_HOURS)
    row = con.execute(
        "SELECT COUNT(*) FROM alerts "
        "WHERE alert_type = ? AND headline = ? AND fired_at > ?",
        [alert.alert_type, alert.headline, cutoff],
    ).fetchone()
    return bool(row and row[0] > 0)


def dispatch(cfg: Config, alerts: Iterable[Alert], sink_dir: Path = Path("alerts")) -> list[dict]:
    out: list[dict] = []
    with connect(cfg.storage.duckdb_path) as con:
        for a in alerts:
            if _is_debounced(con, a):
                out.append({"alert_id": a.alert_id, "skipped": "debounced",
                            "type": a.alert_type})
                continue
            sinks = [_file_sink(a, sink_dir)]
            tg = _telegram_send(a)
            if tg is not None:
                sinks.append(tg)
            channel = ",".join(sinks)
            con.execute(
                "INSERT INTO alerts (alert_id, ts, alert_type, priority, headline, body, fired_at, delivered, channel) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [a.alert_id, a.ts, a.alert_type, a.priority, a.headline, a.body,
                 datetime.utcnow(), True, channel],
            )
            out.append({"alert_id": a.alert_id, "type": a.alert_type,
                        "priority": a.priority, "channels": sinks})
    return out
