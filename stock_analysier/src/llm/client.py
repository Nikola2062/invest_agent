"""DeepSeek LLM client (OpenAI-compatible API).

Two entry points:
  - chat_text(): plain text completion, for free-form analysis
  - chat_json(): structured JSON completion validated against a pydantic model
"""
from __future__ import annotations

import json
import logging
import time
from typing import Type, TypeVar

from openai import OpenAI
from openai import APIError, APIConnectionError, RateLimitError
from pydantic import BaseModel, ValidationError

from src.config.loader import load_secrets

log = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class LLMError(Exception):
    pass


class LLMConfigError(LLMError):
    """API key missing or otherwise unconfigured."""


_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is not None:
        return _client
    secrets = load_secrets()
    if not secrets.deepseek.api_key:
        raise LLMConfigError(
            "DeepSeek API key not set. Edit config/secrets.yaml and paste your key under deepseek.api_key."
        )
    _client = OpenAI(
        api_key=secrets.deepseek.api_key,
        base_url=secrets.deepseek.base_url,
    )
    return _client


def _model_id(reasoner: bool = False) -> str:
    secrets = load_secrets()
    return secrets.deepseek.model_reasoner if reasoner else secrets.deepseek.model_default


def chat_text(
    *,
    system: str,
    user: str,
    reasoner: bool = False,
    temperature: float = 0.3,
    max_retries: int = 3,
) -> str:
    """Plain text completion."""
    client = _get_client()
    model = _model_id(reasoner)
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            )
            return resp.choices[0].message.content or ""
        except (APIConnectionError, RateLimitError, APIError) as e:
            last_err = e
            sleep = 2 ** attempt
            log.warning("LLM call failed (attempt %d/%d): %s. Retrying in %ds.", attempt + 1, max_retries, e, sleep)
            time.sleep(sleep)
    raise LLMError(f"LLM call failed after {max_retries} attempts: {last_err}")


def chat_json(
    *,
    system: str,
    user: str,
    schema: Type[T],
    reasoner: bool = False,
    temperature: float = 0.2,
    max_retries: int = 3,
) -> T:
    """Structured JSON completion validated against a pydantic schema.

    Injects schema instructions into the system prompt and asks DeepSeek for
    response_format=json_object. Parses + validates. Retries on JSON / validation
    errors with an explicit correction message.
    """
    client = _get_client()
    model = _model_id(reasoner)

    schema_json = json.dumps(schema.model_json_schema(), indent=2)
    full_system = (
        f"{system}\n\n"
        f"You MUST respond with a single JSON object that conforms to this schema:\n"
        f"```json\n{schema_json}\n```\n"
        f"Return ONLY the JSON object — no markdown fences, no commentary."
    )

    last_err: Exception | None = None
    correction: str | None = None
    for attempt in range(max_retries):
        try:
            messages = [
                {"role": "system", "content": full_system},
                {"role": "user", "content": user},
            ]
            if correction:
                messages.append({"role": "user", "content": correction})

            resp = client.chat.completions.create(
                model=model,
                temperature=temperature,
                messages=messages,
                response_format={"type": "json_object"},
            )
            raw = resp.choices[0].message.content or ""
            data = json.loads(raw)
            return schema.model_validate(data)
        except json.JSONDecodeError as e:
            last_err = e
            correction = f"Your previous response was not valid JSON: {e}. Return ONLY a valid JSON object."
            log.warning("LLM returned invalid JSON (attempt %d/%d).", attempt + 1, max_retries)
        except ValidationError as e:
            last_err = e
            correction = (
                f"Your previous JSON did not match the required schema. Pydantic errors:\n{e}\n"
                f"Return a corrected JSON object matching the schema exactly."
            )
            log.warning("LLM JSON failed schema validation (attempt %d/%d).", attempt + 1, max_retries)
        except (APIConnectionError, RateLimitError, APIError) as e:
            last_err = e
            sleep = 2 ** attempt
            log.warning("LLM call failed (attempt %d/%d): %s. Retrying in %ds.", attempt + 1, max_retries, e, sleep)
            time.sleep(sleep)
    raise LLMError(f"LLM JSON call failed after {max_retries} attempts: {last_err}")
