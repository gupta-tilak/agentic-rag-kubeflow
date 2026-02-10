"""Optional re-ranking step applied after initial retrieval."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from langchain_core.documents import Document


def rerank(query: str, documents: list[Document], top_k: int = 3) -> list[Document]:
    """Re-rank *documents* for *query* and return the top-*top_k*.

    This is a placeholder for a cross-encoder or Cohere Rerank integration.
    Currently returns documents sorted by their existing relevance score
    metadata (if present) or in the original order.

    TODO: Plug in a cross-encoder model (e.g. ``cross-encoder/ms-marco-MiniLM-L-6-v2``).
    """
    # Placeholder: just truncate to top_k
    return documents[:top_k]
