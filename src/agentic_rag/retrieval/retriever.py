"""Semantic retriever — metadata-aware search with citation tracking.

This module is the **primary public interface** for retrieval.  It is
intentionally decoupled from LangChain retriever abstractions so that
non-agent callers (evaluation scripts, notebooks, tests) can use it
directly.

Usage::

    from agentic_rag.retrieval.retriever import SemanticRetriever

    retriever = SemanticRetriever()
    results   = retriever.search("How does KServe autoscaling work?", k=5)
    for r in results:
        print(r.citation.short_ref(), r.content[:80])
"""

from __future__ import annotations

import logging
from typing import Any

from agentic_rag.retrieval.base import VectorStoreBase
from agentic_rag.retrieval.models import Citation, MetadataFilter, RetrievalResult

logger = logging.getLogger(__name__)


class SemanticRetriever:
    """High-level retriever that wraps any :class:`VectorStoreBase`.

    Parameters
    ----------
    store:
        A concrete vector-store backend.  When *None*, a default
        :class:`~agentic_rag.retrieval.chroma_store.ChromaVectorStore`
        is created from the global settings.
    default_k:
        Default number of results returned by :meth:`search`.
    score_threshold:
        Minimum similarity score; results below this are discarded.
    """

    def __init__(
        self,
        store: VectorStoreBase | None = None,
        *,
        default_k: int = 5,
        score_threshold: float = 0.0,
    ) -> None:
        if store is None:
            from agentic_rag.retrieval.chroma_store import ChromaVectorStore

            store = ChromaVectorStore()
        self._store = store
        self.default_k = default_k
        self.score_threshold = score_threshold

    # -- public API -----------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        k: int | None = None,
        filters: list[MetadataFilter] | None = None,
    ) -> list[RetrievalResult]:
        """Run a semantic search and return results with citations.

        Parameters
        ----------
        query:
            Natural-language query string.
        k:
            Number of results (defaults to ``self.default_k``).
        filters:
            Optional metadata filters forwarded to the vector store.

        Returns
        -------
        list[RetrievalResult]
            Ranked results, each carrying a :class:`Citation`.
        """
        k = k or self.default_k
        raw_hits = self._store.similarity_search_by_text(query, k=k, filters=filters)
        return self._to_results(raw_hits)

    def search_by_embedding(
        self,
        embedding: list[float],
        *,
        k: int | None = None,
        filters: list[MetadataFilter] | None = None,
    ) -> list[RetrievalResult]:
        """Same as :meth:`search` but accepts a pre-computed embedding."""
        k = k or self.default_k
        raw_hits = self._store.similarity_search(embedding, k=k, filters=filters)
        return self._to_results(raw_hits)

    # -- LangChain compat (existing callers) ----------------------------------

    def as_langchain_retriever(self, k: int = 5) -> Any:
        """Return a thin LangChain-compatible retriever wrapper.

        This intentionally imports LangChain only here so that the rest
        of the retrieval package has **zero** LangChain dependency.
        """
        from langchain_core.documents import Document
        from langchain_core.retrievers import BaseRetriever

        outer = self

        class _LCRetriever(BaseRetriever):
            """Adapter that satisfies LangChain's retriever protocol."""

            class Config:  # noqa: D106
                arbitrary_types_allowed = True

            def _get_relevant_documents(self_inner, query: str, **kwargs: Any) -> list[Document]:  # type: ignore[override]  # noqa: N805
                results = outer.search(query, k=k)
                return [
                    Document(
                        page_content=r.content,
                        metadata={**r.citation.metadata, "_citation": r.citation.model_dump()},
                    )
                    for r in results
                ]

        return _LCRetriever()

    # -- internals ------------------------------------------------------------

    def _to_results(self, raw_hits: list[dict[str, Any]]) -> list[RetrievalResult]:
        results: list[RetrievalResult] = []
        for hit in raw_hits:
            score = hit.get("score")
            if score is not None and score < self.score_threshold:
                continue

            meta = hit.get("metadata", {})
            citation = Citation(
                document_id=hit.get("id"),
                source=meta.get("source", "unknown"),
                chunk_index=meta.get("chunk_index"),
                page=meta.get("page"),
                score=score,
                metadata=meta,
            )
            results.append(RetrievalResult(content=hit.get("content", ""), citation=citation))
        return results


# ---------------------------------------------------------------------------
# Convenience factory — backward-compatible with old ``get_retriever()``
# ---------------------------------------------------------------------------


def get_retriever(k: int = 5) -> Any:
    """Return a LangChain-compatible retriever (drop-in replacement).

    Existing code that called ``get_retriever()`` continues to work
    unchanged.
    """
    return SemanticRetriever().as_langchain_retriever(k=k)
