"""YAML config loaders returning typed pydantic models."""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.models.schemas import (
    CompetenceConfig,
    Portfolio,
    Universe,
    RiskPolicy,
    Schedule,
    Secrets,
    TechnicalConfig,
    ValuationConfig,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def _load_yaml(filename: str) -> dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=1)
def load_portfolio() -> Portfolio:
    return Portfolio.model_validate(_load_yaml("portfolio.yaml"))


@lru_cache(maxsize=1)
def load_universe() -> Universe:
    return Universe.model_validate(_load_yaml("universe.yaml"))


@lru_cache(maxsize=1)
def load_risk_policy() -> RiskPolicy:
    return RiskPolicy.model_validate(_load_yaml("risk_policy.yaml"))


@lru_cache(maxsize=1)
def load_schedule() -> Schedule:
    return Schedule.model_validate(_load_yaml("schedule.yaml"))


def _env_overlay_secrets(data: dict[str, Any]) -> dict[str, Any]:
    """Overlay credentials from environment variables onto the secrets dict.

    Env wins over secrets.yaml when set, so credentials can be passed at launch
    (e.g. by the orchestrator) instead of living on disk. Absent env => YAML
    value is used unchanged (backward compatible).
    """
    data = dict(data or {})
    mapping = {
        ("deepseek", "api_key"): "DEEPSEEK_API_KEY",
        ("deepseek", "base_url"): "DEEPSEEK_BASE_URL",
        ("finnhub", "api_key"): "FINNHUB_API_KEY",
        ("telegram", "bot_token"): "TELEGRAM_BOT_TOKEN",
        ("telegram", "chat_id"): "TELEGRAM_CHAT_ID",
        ("polygon", "api_key"): "POLYGON_API_KEY",
        ("newsapi", "api_key"): "NEWSAPI_API_KEY",
        ("fred", "api_key"): "FRED_API_KEY",
    }
    for (section, key), env_name in mapping.items():
        val = os.environ.get(env_name)
        if val:
            if not isinstance(data.get(section), dict):
                data[section] = {}
            data[section][key] = val
    return data


@lru_cache(maxsize=1)
def load_secrets() -> Secrets:
    try:
        data = _load_yaml("secrets.yaml")
    except FileNotFoundError:
        data = {}  # all secrets may come from the environment instead
    return Secrets.model_validate(_env_overlay_secrets(data))


@lru_cache(maxsize=1)
def load_competence() -> CompetenceConfig:
    try:
        return CompetenceConfig.model_validate(_load_yaml("competence.yaml"))
    except FileNotFoundError:
        # If the file is missing, treat everything as borderline (permissive default)
        return CompetenceConfig()


@lru_cache(maxsize=1)
def load_technical() -> TechnicalConfig:
    try:
        return TechnicalConfig.model_validate(_load_yaml("technical.yaml"))
    except FileNotFoundError:
        # Missing file → all defaults (permissive; Technical Division still runs)
        return TechnicalConfig()


@lru_cache(maxsize=1)
def load_valuation() -> ValuationConfig:
    try:
        return ValuationConfig.model_validate(_load_yaml("valuation.yaml"))
    except FileNotFoundError:
        # Missing file → DCF defaults that match the pre-config hardcoded values
        return ValuationConfig()


def reload_all() -> None:
    """Clear caches so the next call re-reads from disk."""
    load_portfolio.cache_clear()
    load_universe.cache_clear()
    load_risk_policy.cache_clear()
    load_schedule.cache_clear()
    load_secrets.cache_clear()
    load_competence.cache_clear()
    load_technical.cache_clear()
    load_valuation.cache_clear()
