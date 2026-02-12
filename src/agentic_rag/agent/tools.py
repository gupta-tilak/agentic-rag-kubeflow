"""LangGraph tool definitions exposed to the agent.

Each tool is a self-contained retrieval capability that the agent can
invoke autonomously.  Tools operate against the existing retrieval stack
(:mod:`agentic_rag.retrieval`) so they work both inside Kubeflow
pipelines and locally during development / testing.

Dependency-injection note
-------------------------
Every tool accepts an *optional* ``retriever`` parameter.  In production
the default ``SemanticRetriever`` is used; in tests a lightweight fake
can be injected instead.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.documents import Document
from langchain_core.tools import tool

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_retriever() -> Any:
    """Lazy-import to avoid import-time side-effects (Chroma connection)."""
    from agentic_rag.retrieval.retriever import SemanticRetriever

    return SemanticRetriever()


def _results_to_documents(results: list[Any]) -> list[Document]:
    """Convert :class:`RetrievalResult` objects to LangChain Documents.

    The full citation payload is preserved inside ``metadata["_citation"]``.
    """
    docs: list[Document] = []
    for r in results:
        docs.append(
            Document(
                page_content=r.content,
                metadata={
                    "source": r.citation.source,
                    "chunk_index": r.citation.chunk_index,
                    "score": r.citation.score,
                    "citation_id": r.citation.citation_id,
                    **r.citation.metadata,
                },
            )
        )
    return docs


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool
def vector_search(query: str, k: int = 5) -> list[Document]:
    """Perform a semantic (embedding-based) search over the knowledge base.

    Use this tool when the user asks a conceptual or open-ended question
    and you need passages that are **semantically** close to the query.

    Parameters
    ----------
    query:
        Natural-language search query.
    k:
        Number of results to return (default 5).

    Returns
    -------
    list[Document]
        Top-k documents with citation metadata.
    """
    retriever = _get_retriever()
    results = retriever.search(query, k=k)
    logger.info("vector_search returned %d results for %r", len(results), query)
    return _results_to_documents(results)


@tool
def keyword_search(query: str, source_filter: str | None = None, k: int = 5) -> list[Document]:
    """Perform a filtered search, optionally restricting to a specific source.

    Use this tool when the user mentions a **specific document, file, or
    topic** and you want to restrict retrieval to that source.

    Parameters
    ----------
    query:
        Natural-language search query.
    source_filter:
        If provided, limit results to documents whose ``source`` metadata
        matches this value (exact match).
    k:
        Number of results to return (default 5).

    Returns
    -------
    list[Document]
        Matching documents with citation metadata.
    """
    from agentic_rag.retrieval.models import MetadataFilter

    retriever = _get_retriever()
    filters = None
    if source_filter:
        filters = [MetadataFilter.equals("source", source_filter)]
    results = retriever.search(query, k=k, filters=filters)
    logger.info(
        "keyword_search returned %d results for %r (source=%s)",
        len(results),
        query,
        source_filter,
    )
    return _results_to_documents(results)


@tool
def document_lookup(source: str, k: int = 10) -> list[Document]:
    """Retrieve all chunks belonging to a specific source document.

    Use this tool when the user asks about a **particular document** by
    name/path and you want to pull all of its chunks to get the big picture.

    Parameters
    ----------
    source:
        The source identifier (file path, URL, etc.) as stored in metadata.
    k:
        Max chunks to return (default 10).

    Returns
    -------
    list[Document]
        Chunks from the requested source.
    """
    from agentic_rag.retrieval.models import MetadataFilter

    retriever = _get_retriever()
    # Using a broad query with a tight source filter fetches all chunks.
    filters = [MetadataFilter.equals("source", source)]
    results = retriever.search(source, k=k, filters=filters)
    logger.info("document_lookup returned %d chunks for source=%r", len(results), source)
    return _results_to_documents(results)


@tool
def web_search(query: str) -> list[Document]:
    """Search the web for recent or external information.

    **Stub** — returns a placeholder message.  Wire this to a real web
    search API (Tavily, SerpAPI, Brave Search, …) in production.

    Parameters
    ----------
    query:
        Natural-language search query.

    Returns
    -------
    list[Document]
        Web search results (currently a placeholder).
    """
    logger.warning("web_search called but not implemented — returning stub for %r", query)
    return [
        Document(
            page_content=(
                f"[Web search stub] No live web results available for: {query}. "
                "In production, connect a web search provider (Tavily, SerpAPI, etc.)."
            ),
            metadata={"source": "web_search_stub", "query": query},
        )
    ]


# ---------------------------------------------------------------------------
# Tool registry — used by nodes to discover available tools.
# ---------------------------------------------------------------------------

TOOL_REGISTRY: dict[str, Any] = {
    "vector_search": vector_search,
    "keyword_search": keyword_search,
    "document_lookup": document_lookup,
    "web_search": web_search,
}
"""Mapping of tool name → tool callable.  Nodes iterate over this to
execute the retrieval plan."""
