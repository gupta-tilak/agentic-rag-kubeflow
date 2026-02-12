"""
Retrieval — vector search, re-ranking, and context assembly.

This module wraps the vector store behind a clean interface so that
the agent layer never needs to know which DB is backing retrieval.

Public surface
--------------
- :class:`SemanticRetriever` — main entry point for retrieval with citations.
- :class:`VectorStoreBase` — abstract backend (subclass for Pinecone, etc.).
- :class:`ChromaVectorStore` — default Chroma backend.
- :class:`Citation`, :class:`RetrievalResult`, :class:`MetadataFilter` — data models.
- :func:`get_retriever` — backward-compatible LangChain retriever factory.
"""

from agentic_rag.retrieval.base import VectorStoreBase
from agentic_rag.retrieval.models import Citation, MetadataFilter, RetrievalResult
from agentic_rag.retrieval.retriever import SemanticRetriever, get_retriever

__all__ = [
    "Citation",
    "ChromaVectorStore",
    "MetadataFilter",
    "RetrievalResult",
    "SemanticRetriever",
    "VectorStoreBase",
    "get_retriever",
]


def __getattr__(name: str):  # noqa: ANN001
    """Lazy-import ChromaVectorStore to avoid pulling in chromadb at import time."""
    if name == "ChromaVectorStore":
        from agentic_rag.retrieval.chroma_store import ChromaVectorStore

        return ChromaVectorStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
