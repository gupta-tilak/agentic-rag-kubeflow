"""Chroma implementation of the vector-store abstraction."""

from __future__ import annotations

import logging
from typing import Any

import chromadb
from langchain_huggingface import HuggingFaceEmbeddings

from agentic_rag.config import settings
from agentic_rag.retrieval.base import VectorStoreBase
from agentic_rag.retrieval.models import MetadataFilter

logger = logging.getLogger(__name__)


def _build_chroma_where(filters: list[MetadataFilter]) -> dict[str, Any] | None:
    """Convert a list of :class:`MetadataFilter` to Chroma ``where`` syntax."""
    if not filters:
        return None

    _OP_MAP = {
        "eq": "$eq",
        "ne": "$ne",
        "gt": "$gt",
        "gte": "$gte",
        "lt": "$lt",
        "lte": "$lte",
        "in": "$in",
        "nin": "$nin",
    }

    clauses: list[dict[str, Any]] = []
    for f in filters:
        chroma_op = _OP_MAP.get(f.operator)
        if chroma_op is None:
            raise ValueError(f"Unsupported filter operator: {f.operator!r}")
        clauses.append({f.field: {chroma_op: f.value}})

    if len(clauses) == 1:
        return clauses[0]
    return {"$and": clauses}


class ChromaVectorStore(VectorStoreBase):
    """Chroma-backed vector store.

    Parameters
    ----------
    collection_name:
        Name of the Chroma collection.
    host:
        Chroma server hostname.
    port:
        Chroma server port.
    embedding_model:
        HuggingFace model id used for text → embedding conversion.
    """

    def __init__(
        self,
        collection_name: str = settings.chroma_collection,
        *,
        host: str = settings.chroma_host,
        port: int = settings.chroma_port,
        embedding_model: str = settings.embedding_model,
    ) -> None:
        super().__init__(collection_name)
        self._host = host
        self._port = port
        self._client = chromadb.HttpClient(host=host, port=port)
        self._collection = self._client.get_or_create_collection(collection_name)
        self._embedder = HuggingFaceEmbeddings(model_name=embedding_model)

    # -- VectorStoreBase overrides --------------------------------------------

    def similarity_search(
        self,
        query_embedding: list[float],
        *,
        k: int = 5,
        filters: list[MetadataFilter] | None = None,
    ) -> list[dict[str, Any]]:
        where = _build_chroma_where(filters) if filters else None

        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits: list[dict[str, Any]] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc_id, content, meta, dist in zip(ids, docs, metas, distances):
            # Chroma returns L2 distances; convert to a 0-1 similarity score.
            score = 1.0 / (1.0 + dist)
            hits.append(
                {
                    "id": doc_id,
                    "content": content or "",
                    "score": score,
                    "metadata": meta or {},
                }
            )
        return hits

    def similarity_search_by_text(
        self,
        query: str,
        *,
        k: int = 5,
        filters: list[MetadataFilter] | None = None,
    ) -> list[dict[str, Any]]:
        embedding = self._embedder.embed_query(query)
        return self.similarity_search(embedding, k=k, filters=filters)

    def health_check(self) -> bool:
        try:
            self._client.heartbeat()
            return True
        except Exception:
            logger.warning("Chroma health-check failed", exc_info=True)
            return False

    def delete(self, ids: list[str]) -> None:
        self._collection.delete(ids=ids)
