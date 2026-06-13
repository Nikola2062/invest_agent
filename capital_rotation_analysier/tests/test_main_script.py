"""Tests for main.py — credential resolution + failure semantics.

main.py is loaded via importlib because it lives at the repo root (not inside
the rotation package) and isn't on the test's import path by default.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("main_module", ROOT / "main.py")
run_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_module)


# ---------- credential resolution ----------

def test_cli_args_win_over_dotenv():
    cli = {
        "TELEGRAM_BOT_TOKEN": "from-cli",
        "TELEGRAM_CHAT_ID":   "from-cli",
        "DEEPSEEK_API_KEY":   "from-cli",
        "FRED_API_KEY":       "from-cli",
    }
    dotenv = {
        "TELEGRAM_BOT_TOKEN": "from-dotenv",
        "TELEGRAM_CHAT_ID":   "from-dotenv",
        "DEEPSEEK_API_KEY":   "from-dotenv",
        "FRED_API_KEY":       "from-dotenv",
    }
    resolved, missing = run_module.resolve_credentials(cli, dotenv)
    assert missing == []
    assert all(v == "from-cli" for v in resolved.values())


def test_dotenv_used_when_cli_arg_missing():
    cli = {
        "TELEGRAM_BOT_TOKEN": None,
        "TELEGRAM_CHAT_ID":   None,
        "DEEPSEEK_API_KEY":   None,
        "FRED_API_KEY":       None,
    }
    dotenv = {
        "TELEGRAM_BOT_TOKEN": "from-dotenv",
        "TELEGRAM_CHAT_ID":   "from-dotenv",
        "DEEPSEEK_API_KEY":   "from-dotenv",
        "FRED_API_KEY":       "from-dotenv",
    }
    resolved, missing = run_module.resolve_credentials(cli, dotenv)
    assert missing == []
    assert resolved["TELEGRAM_BOT_TOKEN"] == "from-dotenv"


def test_mixed_sources():
    cli = {
        "TELEGRAM_BOT_TOKEN": "cli-token",
        "TELEGRAM_CHAT_ID":   None,        # fall through to dotenv
        "DEEPSEEK_API_KEY":   "cli-ds",
        "FRED_API_KEY":       None,        # fall through to dotenv
    }
    dotenv = {
        "TELEGRAM_CHAT_ID":   "env-chat",
        "FRED_API_KEY":       "env-fred",
    }
    resolved, missing = run_module.resolve_credentials(cli, dotenv)
    assert missing == []
    assert resolved["TELEGRAM_BOT_TOKEN"] == "cli-token"
    assert resolved["TELEGRAM_CHAT_ID"]   == "env-chat"
    assert resolved["DEEPSEEK_API_KEY"]   == "cli-ds"
    assert resolved["FRED_API_KEY"]       == "env-fred"


def test_fail_when_anything_missing():
    cli = {k: None for k, _, _ in run_module.REQUIRED_KEYS}
    dotenv: dict[str, str] = {}
    resolved, missing = run_module.resolve_credentials(cli, dotenv)
    assert resolved == {}
    assert len(missing) == 4
    # All four keys should be reported
    env_names = {m[0] for m in missing}
    assert env_names == {"TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID",
                          "DEEPSEEK_API_KEY", "FRED_API_KEY"}


def test_partial_missing_only_names_offenders():
    """If three are supplied and one isn't, only the missing one appears."""
    cli = {
        "TELEGRAM_BOT_TOKEN": "ok",
        "TELEGRAM_CHAT_ID":   "ok",
        "DEEPSEEK_API_KEY":   "ok",
        "FRED_API_KEY":       None,        # missing
    }
    dotenv: dict[str, str] = {}
    _, missing = run_module.resolve_credentials(cli, dotenv)
    assert len(missing) == 1
    assert missing[0][0] == "FRED_API_KEY"


# ---------- .env file parsing ----------

def test_load_dotenv_handles_quotes_and_comments(tmp_path):
    p = tmp_path / ".env"
    p.write_text(
        "# A comment line\n"
        "\n"
        "TELEGRAM_BOT_TOKEN=plain-value\n"
        'TELEGRAM_CHAT_ID="quoted-value"\n'
        "DEEPSEEK_API_KEY='single-quoted'\n"
        "  FRED_API_KEY=  spaced  \n"        # surrounding whitespace
        "MALFORMED_LINE_NO_EQUALS\n"          # ignored
    )
    out = run_module.load_dotenv(str(p))
    assert out["TELEGRAM_BOT_TOKEN"] == "plain-value"
    assert out["TELEGRAM_CHAT_ID"]   == "quoted-value"
    assert out["DEEPSEEK_API_KEY"]   == "single-quoted"
    assert out["FRED_API_KEY"]       == "spaced"
    assert "MALFORMED_LINE_NO_EQUALS" not in out


def test_load_dotenv_returns_empty_when_file_missing(tmp_path):
    out = run_module.load_dotenv(str(tmp_path / "does-not-exist.env"))
    assert out == {}


# ---------- error message ----------

def test_missing_error_message_lists_offenders():
    missing = [
        ("TELEGRAM_BOT_TOKEN",  "--telegram-token",   "Telegram bot token"),
        ("FRED_API_KEY",         "--fred-key",        "FRED API key (RRP / yield curve)"),
    ]
    msg = run_module._format_missing_error(missing)
    assert "TELEGRAM_BOT_TOKEN" in msg
    assert "--telegram-token"   in msg
    assert "FRED_API_KEY"       in msg
    assert "--fred-key"         in msg
    # Should NOT mention keys that weren't missing
    assert "DEEPSEEK_API_KEY"   not in msg


# ---------- exit-2 semantics via main() ----------

def test_main_exits_2_when_creds_missing(monkeypatch, tmp_path, capsys):
    # Point SCRIPT_DIR at an empty tmpdir so the .env lookup finds nothing.
    # (Without this the project root's real .env would be picked up.)
    monkeypatch.setattr(run_module, "SCRIPT_DIR", tmp_path)
    rc = run_module.main(argv=[])
    assert rc == 2
    err = capsys.readouterr().err
    assert "Missing required credentials" in err
