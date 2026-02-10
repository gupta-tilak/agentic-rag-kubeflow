"""Retriever — wraps vector-store search behind a LangChain retriever."""

from __future__ import annotations

import chromadb
from langchain_community.vectorstores import Chroma
from langchain_core.retrievers import BaseRetriever

from agentic_rag.config import settings
from agentic_rag.ingestion.embedder import get_embedding_function


def get_vectorstore() -> Chroma:
    """Connect to the running Chroma instance and return the vector store."""
    client = chromadb.HttpClient(host=settings.chroma_host, port=settings.chroma_port)
    return Chroma(
        client=client,
        collection_name=settings.chroma_collection,
        embedding_function=get_embedding_function(),
    )


def get_retriever(k: int = 5) -> BaseRetriever:
    """Return a LangChain-compatible retriever over the default collection.

    Parameters
    ----------
    k:
        Number of documents to retrieve per query.
    """
    vectorstore = get_vectorstore()
    return vectorstore.as_retriever(search_kwargs={"k": k})
