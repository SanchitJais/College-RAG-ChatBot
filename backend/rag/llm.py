"""One place that talks to the chat model. Works with any OpenAI-compatible endpoint
(Groq and Gemini free tiers, local Ollama, ...), configured in .env."""
import json
import logging
from functools import lru_cache

from openai import OpenAI, OpenAIError, RateLimitError

from backend.config import settings

log = logging.getLogger(__name__)


class GenerationError(Exception):
    """The LLM failed or returned unusable output. A service error, not missing knowledge."""


class RateLimitedError(GenerationError):
    """The (free-tier) model is rate limited; retrying in a few seconds usually works."""


@lru_cache
def _client() -> OpenAI:
    # Local servers (Ollama) ignore the key but the SDK requires a non-empty one.
    return OpenAI(base_url=settings.llm_base_url, api_key=settings.llm_api_key or "not-needed",
                  max_retries=settings.llm_max_retries, timeout=60)


def call_llm(system: str, user: str) -> str:
    """Single chat call; returns the raw reply text. (Tests replace this function.)"""
    try:
        resp = _client().chat.completions.create(
            model=settings.llm_model,
            temperature=0,
            max_tokens=1500,  # reasoning models spend tokens thinking before they answer
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
    except RateLimitError as exc:
        raise RateLimitedError("Language model is rate limited.") from exc
    except OpenAIError as exc:
        status = getattr(exc, "status_code", None)
        hint = {
            401: "the API key was rejected - check LLM_API_KEY in .env (it may be revoked)",
            403: "access denied - check LLM_API_KEY and that the key can use this model",
            404: "model or URL not found - check LLM_MODEL and LLM_BASE_URL in .env",
        }.get(status, "see the status above")
        log.error("LLM request failed: %s (HTTP %s): %s", type(exc).__name__, status, hint)
        # Only the exception type is exposed to callers: messages can echo request details.
        raise GenerationError(f"Language model request failed ({type(exc).__name__}).") from exc
    return resp.choices[0].message.content or ""


def parse_json_object(raw: str) -> dict:
    """Parse the JSON object in a model reply, tolerating ```json fences or chatter around it."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object in reply")
    data = json.loads(raw[start : end + 1])
    if not isinstance(data, dict):
        raise ValueError("reply is not a JSON object")
    return data
