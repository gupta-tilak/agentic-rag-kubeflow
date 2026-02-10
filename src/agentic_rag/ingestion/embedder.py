"""Embedding and vector-store persistence."""

from __future__ import annotations

from typing import TYPE_CHECKING

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from agentic_rag.config import settings

if TYPE_CHECKING:
    from langchain_core.documents import Document


def get_embedding_function() -> HuggingFaceEmbeddings:
    """Return the configured sentence-transformer embedding function."""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model)


def embed_and_store(documents: list[Document]) -> Chroma:
    """Embed *documents* and upsert them into the Chroma collection.

    Returns the ``Chroma`` vector-store handle so callers can query it
    immediately if needed.
    """
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=get_embedding_function(),
        client=client,
        collection_name=settings.chroma_collection,
    )
    return vectorstore
