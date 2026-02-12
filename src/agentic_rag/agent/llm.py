"""LLM initialisation — single place to swap providers.

Supports two modes:

1. **OpenAI cloud** (default) — set ``OPENAI_API_KEY``.
2. **KServe vLLM endpoint** — set ``LLM_BASE_URL`` to the in-cluster
   vLLM service (e.g. ``http://llm-server.kubeflow-user.svc.cluster.local/v1``).
   The vLLM runtime exposes an OpenAI-compatible ``/v1/chat/completions``
   endpoint, so ``ChatOpenAI`` works unchanged.
"""

from __future__ import annotations

import logging

from langchain_openai import ChatOpenAI

from agentic_rag.config import settings

logger = logging.getLogger(__name__)


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Return the configured chat model.

    When ``settings.llm_base_url`` is set the client is pointed at the
    KServe vLLM InferenceService instead of the OpenAI cloud API.
    A dummy API key (``"EMPTY"``) is used because vLLM does not
    require authentication.
    """
    kwargs: dict = {
        "model": settings.llm_model_name,
        "temperature": temperature,
    }

    if settings.llm_base_url:
        logger.info("Using KServe vLLM endpoint: %s", settings.llm_base_url)
        kwargs["base_url"] = settings.llm_base_url
        # vLLM doesn't need a real key; LangChain requires a non-empty value.
        kwargs["api_key"] = settings.openai_api_key or "EMPTY"
    else:
        kwargs["api_key"] = settings.openai_api_key

    return ChatOpenAI(**kwargs)
