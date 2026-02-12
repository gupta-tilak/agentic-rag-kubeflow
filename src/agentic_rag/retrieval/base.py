"""Abstract base class for vector-store backends.

Adding a new backend (Pinecone, Weaviate, Qdrant …) only requires
subclassing :class:`VectorStoreBase` and implementing the four abstract
methods.  The rest of the retrieval stack is backend-agnostic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from agentic_rag.retrieval.models import MetadataFilter


class VectorStoreBase(ABC):
    """Backend-agnostic vector-store interface.

    Parameters
    ----------
    collection_name:
        Logical name of the collection / index / namespace.
    """

    def __init__(self, collection_name: str) -> None:
        self.collection_name = collection_name

    # -- required overrides ---------------------------------------------------

    @abstractmethod
    def similarity_search(
        self,
        query_embedding: list[float],
        *,
        k: int = 5,
        filters: list[MetadataFilter] | None = None,
    ) -> list[dict[str, Any]]:
        """Return the top-*k* results matching *query_embedding*.

        Each result dict **must** contain at least:

        * ``"id"`` – document / chunk identifier
        * ``"content"`` – the textual content
        * ``"score"`` – similarity score (higher = more similar)
        * ``"metadata"`` – associated metadata dict

        Parameters
        ----------
        query_embedding:
            Dense vector for the query.
        k:
            Number of results to return.
        filters:
            Optional metadata filters applied server-side.
        """
        ...

    @abstractmethod
    def similarity_search_by_text(
        self,
        query: str,
        *,
        k: int = 5,
        filters: list[MetadataFilter] | None = None,
    ) -> list[dict[str, Any]]:
        """Embed *query* internally and delegate to :meth:`similarity_search`.

        Backends that accept raw text queries can override this for
        efficiency; the default implementation embeds the query first.
        """
        ...

    @abstractmethod
    def health_check(self) -> bool:
        """Return ``True`` when the backend is reachable and ready."""
        ...

    # -- optional overrides ---------------------------------------------------

    def delete(self, ids: list[str]) -> None:
        """Delete documents by their IDs.  Optional — raises by default."""
        raise NotImplementedError(f"{type(self).__name__} does not support delete")
