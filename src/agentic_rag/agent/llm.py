"""LLM initialisation — single place to swap providers."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

from agentic_rag.config import settings


def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Return the configured chat model.

    Swap this function body to switch from OpenAI to a local vLLM,
    Ollama, or any other LangChain-compatible chat model.
    """
    return ChatOpenAI(
        model=settings.llm_model_name,
        temperature=temperature,
        api_key=settings.openai_api_key,
    )
