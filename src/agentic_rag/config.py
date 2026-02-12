"""Shared configuration loaded from environment / YAML."""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide settings, populated from env vars or .env file."""

    # LLM
    openai_api_key: str = Field(default="", description="OpenAI API key (or dummy value for local vLLM)")
    llm_model_name: str = Field(default="gpt-4o-mini", description="LLM model identifier")
    llm_base_url: str = Field(
        default="",
        description=(
            "Base URL for the LLM API. Leave empty to use OpenAI cloud. "
            "Set to the KServe vLLM endpoint for local serving, e.g. "
            "'http://llm-server.kubeflow-user.svc.cluster.local/v1'"
        ),
    )

    # Vector store
    chroma_host: str = "localhost"
    chroma_port: int = 8000
    chroma_collection: str = "agentic_rag"

    # Embedding
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Serving
    kserve_inference_endpoint: str = "http://localhost:8080/v2/models/rag-agent/infer"

    # Kubeflow
    kfp_host: str = "http://localhost:8888"
    kfp_namespace: str = "kubeflow-user"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


# Singleton — import `settings` wherever needed.
settings = Settings()
