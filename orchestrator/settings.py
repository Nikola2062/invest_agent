"""Central settings: paths, config.yaml, and .env loading for the orchestrator.

The orchestrator is a NON-DESTRUCTIVE overlay. It never imports the three
projects' code into this process; it drives them through their own venvs
(subprocess) and reads their on-disk data stores. All paths resolve relative
to the investor_agent root (the parent of this orchestrator/ directory).
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ORCH_DIR = Path(__file__).resolve().parent
ROOT = ORCH_DIR.parent  # .../investor_agent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency). Real env wins via setdefault."""
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ORCH_DIR / ".env")


def load_config() -> dict:
    with open(ORCH_DIR / "config.yaml") as f:
        return yaml.safe_load(f)


CONFIG = load_config()


def project_dir(key: str) -> Path:
    """Absolute path to one of the three projects (key: macro_risk|rotation|stocks)."""
    return (ROOT / CONFIG["projects"][key]["dir"]).resolve()


def project_python(key: str | None = None) -> Path:
    """The single shared interpreter at the repo-root .venv.

    All projects are driven with this one interpreter (their third-party deps are
    installed once at the root); their own source is imported from their folder.
    `key` is accepted for backward compatibility but ignored.
    """
    return ROOT / ".venv" / "bin" / "python"


# --- Telegram (secrets from env / .env, never committed) -------------------
TELEGRAM_BOT_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID", "")


def allowlist_chat_ids() -> list[str]:
    """Chat ids permitted to trigger on-demand (Mode B). Config wins; else the
    single configured chat id from env."""
    ids = [str(x) for x in (CONFIG.get("telegram", {}).get("allowlist_chat_ids") or [])]
    if not ids and TELEGRAM_CHAT_ID:
        ids = [TELEGRAM_CHAT_ID]
    return ids
