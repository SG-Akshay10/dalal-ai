"""LLM Provider abstraction — supports OpenAI, Gemini, and Sarvam AI.

Provider is selected via the LLM_PROVIDER environment variable.
All providers expose the same interface: `get_llm_client()` returns a client
with a `.chat()` method compatible with LangChain's ChatModel interface.

Used by Phase 2 agents (fundamental_agent, sentiment_agent, report_agent, etc.)
"""
import logging
import os
from enum import StrEnum
from typing import Any

logger = logging.getLogger(__name__)


class LLMProvider(StrEnum):
    """Supported LLM providers."""
    OPENAI = "openai"
    GEMINI = "gemini"
    SARVAM = "sarvam"


def get_provider() -> LLMProvider:
    """Read the LLM_PROVIDER env var and return the selected provider enum."""
    raw = os.getenv("LLM_PROVIDER", "openai").lower().strip()
    try:
        return LLMProvider(raw)
    except ValueError:
        logger.warning(
            "Unknown LLM_PROVIDER '%s' — falling back to 'openai'. "
            "Valid options: openai, gemini, sarvam",
            raw,
        )
        return LLMProvider.OPENAI


def get_llm_client() -> Any:
    """Return a LangChain-compatible ChatModel for the configured provider.

    Environment variables required per provider:
    - openai:  OPENAI_API_KEY, OPENAI_MODEL (default: gpt-4o)
    - gemini:  GEMINI_API_KEY, GEMINI_MODEL (default: gemini-2.0-flash)
    - sarvam:  SARVAM_API_KEY, SARVAM_MODEL (default: sarvam-105b),
               SARVAM_BASE_URL (default: https://api.sarvam.ai/v1)

    Returns:
        A ChatModel instance (ChatOpenAI, ChatGoogleGenerativeAI, or
        ChatOpenAI with custom base_url for Sarvam).

    Raises:
        ValueError: If the required API key is not set.
    """
    provider = get_provider()

    if provider == LLMProvider.OPENAI:
        return _build_openai_client()
    elif provider == LLMProvider.GEMINI:
        return _build_gemini_client()
    elif provider == LLMProvider.SARVAM:
        return _build_sarvam_client()
    else:
        raise ValueError(f"Unsupported LLM provider: {provider}")


def _build_openai_client():
    """Build a LangChain ChatOpenAI client."""
    from langchain_openai import ChatOpenAI

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")

    model = os.getenv("OPENAI_MODEL", "gpt-4o")
    logger.info("Using OpenAI provider: model=%s", model)

    return ChatOpenAI(
        model=model,
        api_key=api_key,
        temperature=0.3,
    )


def _build_gemini_client():
    """Build a LangChain ChatGoogleGenerativeAI client."""
    from langchain_google_genai import ChatGoogleGenerativeAI

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is required when LLM_PROVIDER=gemini")

    model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
    logger.info("Using Gemini provider: model=%s", model)

    return ChatGoogleGenerativeAI(
        model=model,
        google_api_key=api_key,
        temperature=0.3,
    )


def _build_sarvam_client():
    """Build a LangChain ChatOpenAI client pointing at Sarvam's OpenAI-compatible endpoint.

    Sarvam AI exposes an OpenAI-compatible /chat/completions endpoint,
    so we reuse ChatOpenAI with a custom base_url.
    """
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
