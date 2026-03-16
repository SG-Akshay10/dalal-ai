"""LLM Provider abstraction — currently supports Sarvam AI only.

Provider is selected via the LLM_PROVIDER environment variable or auto-detected
from available API keys. At present, the only valid provider value is "sarvam";
any other value will be ignored and Sarvam will be used as the fallback.

All providers (currently just Sarvam) expose the same interface:
`get_llm_client()` returns a client with a `.chat()` method compatible with
LangChain's ChatModel interface.

Used by Phase 2 agents (fundamental_agent, sentiment_agent, report_agent, etc.)
"""
import logging
import os
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class LLMProvider(StrEnum):
    """Supported LLM providers."""
    SARVAM = "sarvam"


def get_provider(preferred_provider: str = None) -> LLMProvider:
    """Determine the provider to use.
    If preferred_provider is passed (e.g. from UI), use it.
    Otherwise read LLM_PROVIDER env var or auto-detect based on API keys.
    """
    raw = (preferred_provider or os.getenv("LLM_PROVIDER", "")).lower().strip()

    # If a provider is explicitly requested, validate it and fail fast if unsupported.
    if raw:
        try:
            return LLMProvider(raw)
        except ValueError as exc:
            # Non-sarvam providers are not supported; avoid silently falling back.
            raise ValueError(f"Unsupported LLM provider: {raw}") from exc

    # Auto-detection based on priority: Sarvam
    if os.getenv("SARVAM_API_KEY"):
        return LLMProvider.SARVAM

    # Default fallback
    return LLMProvider.SARVAM


def get_llm_client(preferred_provider: str = None) -> Any:
    """Return a LangChain-compatible ChatModel for the configured provider.

    Args:
        preferred_provider: Optional string ("sarvam") to override defaults.

    Returns:
        A ChatModel instance (ChatOpenAI for Sarvam).
    """
    provider = get_provider(preferred_provider)

    if provider == LLMProvider.SARVAM:
        return _build_sarvam_client()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _build_sarvam_client():
    """Build a LangChain ChatOpenAI client pointing at Sarvam's OpenAI-compatible endpoint."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        raise ValueError("SARVAM_API_KEY is required when LLM_PROVIDER=sarvam")

    model = os.getenv("SARVAM_MODEL", "sarvam-105b")
    base_url = os.getenv("SARVAM_BASE_URL", "https://api.sarvam.ai/v1")
    logger.info("Using Sarvam AI provider: model=%s, base_url=%s", model, base_url)

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=0.3,
    )
