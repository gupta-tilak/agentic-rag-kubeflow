"""Unit tests for the retrieval layer — models, base, and SemanticRetriever."""

from __future__ import annotations

from typing import Any

import pytest

from agentic_rag.retrieval.base import VectorStoreBase
from agentic_rag.retrieval.models import Citation, MetadataFilter, RetrievalResult
from agentic_rag.retrieval.retriever import SemanticRetriever


# ── Fake vector store for deterministic testing ─────────────────────────


class FakeVectorStore(VectorStoreBase):
    """In-memory fake that returns canned results."""

    def __init__(self, hits: list[dict[str, Any]] | None = None) -> None:
        super().__init__("test-collection")
        self._hits: list[dict[str, Any]] = hits or []
        self.last_filters: list[MetadataFilter] | None = None

    def similarity_search(
        self,
        query_embedding: list[float],
        *,
        k: int = 5,
        filters: list[MetadataFilter] | None = None,
    ) -> list[dict[str, Any]]:
        self.last_filters = filters
        return self._hits[:k]

    def similarity_search_by_text(
        self,
        query: str,
        *,
        k: int = 5,
        filters: list[MetadataFilter] | None = None,
    ) -> list[dict[str, Any]]:
        self.last_filters = filters
        return self._hits[:k]

    def health_check(self) -> bool:
        return True


# ── Fixtures ────────────────────────────────────────────────────────────

SAMPLE_HITS: list[dict[str, Any]] = [
    {
        "id": "doc-001",
        "content": "Kubeflow Pipelines orchestrate ML workflows.",
        "score": 0.92,
        "metadata": {"source": "kubeflow_guide.md", "chunk_index": 3, "page": 7},
    },
    {
        "id": "doc-002",
        "content": "KServe provides serverless inference on Kubernetes.",
        "score": 0.87,
        "metadata": {"source": "kserve_docs.md", "chunk_index": 1},
    },
    {
        "id": "doc-003",
        "content": "ChromaDB is an open-source vector database.",
        "score": 0.45,
        "metadata": {"source": "chroma_overview.md"},
    },
]


@pytest.fixture()
def fake_store() -> FakeVectorStore:
    return FakeVectorStore(hits=SAMPLE_HITS)


@pytest.fixture()
def retriever(fake_store: FakeVectorStore) -> SemanticRetriever:
    return SemanticRetriever(store=fake_store, default_k=5)


# ── Citation model tests ───────────────────────────────────────────────


class TestCitation:
    def test_short_ref_with_chunk(self) -> None:
        c = Citation(source="guide.md", chunk_index=3)
        assert c.short_ref() == "[guide.md§3]"

    def test_short_ref_without_chunk(self) -> None:
        c = Citation(source="guide.md")
        assert c.short_ref() == "[guide.md§?]"

    def test_default_source_is_unknown(self) -> None:
        c = Citation()
        assert c.source == "unknown"

    def test_citation_id_is_populated(self) -> None:
        c = Citation()
        assert len(c.citation_id) == 12

    def test_retrieved_at_is_set(self) -> None:
        c = Citation()
        assert c.retrieved_at is not None

    def test_round_trip_serialization(self) -> None:
        c = Citation(source="file.md", chunk_index=2, score=0.9, metadata={"author": "alice"})
        data = c.model_dump()
        restored = Citation(**data)
        assert restored.source == c.source
        assert restored.metadata == c.metadata


# ── MetadataFilter tests ───────────────────────────────────────────────


class TestMetadataFilter:
    def test_equals_factory(self) -> None:
        f = MetadataFilter.equals("source", "guide.md")
        assert f.field == "source"
        assert f.operator == "eq"
        assert f.value == "guide.md"

    def test_not_equals_factory(self) -> None:
        f = MetadataFilter.not_equals("author", "bob")
        assert f.operator == "ne"

    def test_one_of_factory(self) -> None:
        f = MetadataFilter.one_of("source", ["a.md", "b.md"])
        assert f.operator == "in"
        assert f.value == ["a.md", "b.md"]


# ── RetrievalResult tests ─────────────────────────────────────────────


class TestRetrievalResult:
    def test_str_includes_ref_and_content(self) -> None:
        r = RetrievalResult(
            content="Some long content about ML workflows.",
            citation=Citation(source="guide.md", chunk_index=1),
        )
        text = str(r)
        assert "[guide.md§1]" in text
        assert "ML workflows" in text


# ── SemanticRetriever tests ────────────────────────────────────────────


class TestSemanticRetriever:
    def test_search_returns_retrieval_results(self, retriever: SemanticRetriever) -> None:
        results = retriever.search("What is Kubeflow?")
        assert len(results) == 3
        assert all(isinstance(r, RetrievalResult) for r in results)

    def test_citations_populated(self, retriever: SemanticRetriever) -> None:
        results = retriever.search("Kubeflow")
        first = results[0].citation
        assert first.document_id == "doc-001"
        assert first.source == "kubeflow_guide.md"
        assert first.chunk_index == 3
        assert first.page == 7
        assert first.score == 0.92

    def test_k_limits_results(self, fake_store: FakeVectorStore) -> None:
        retriever = SemanticRetriever(store=fake_store, default_k=2)
        results = retriever.search("anything")
        assert len(results) == 2

    def test_explicit_k_overrides_default(self, retriever: SemanticRetriever) -> None:
        results = retriever.search("anything", k=1)
        assert len(results) == 1

    def test_score_threshold_filters(self, fake_store: FakeVectorStore) -> None:
        retriever = SemanticRetriever(store=fake_store, score_threshold=0.5)
        results = retriever.search("query")
        # doc-003 has score=0.45 → should be excluded
        assert len(results) == 2
        assert all(r.citation.score is not None and r.citation.score >= 0.5 for r in results)

    def test_metadata_filters_forwarded(
        self, fake_store: FakeVectorStore, retriever: SemanticRetriever
    ) -> None:
        f = MetadataFilter.equals("source", "kubeflow_guide.md")
        retriever.search("Kubeflow", filters=[f])
        assert fake_store.last_filters is not None
        assert fake_store.last_filters[0].field == "source"

    def test_search_by_embedding(self, retriever: SemanticRetriever) -> None:
        results = retriever.search_by_embedding([0.1, 0.2, 0.3], k=2)
        assert len(results) == 2

    def test_empty_store_returns_empty(self) -> None:
        store = FakeVectorStore(hits=[])
        retriever = SemanticRetriever(store=store)
        assert retriever.search("anything") == []

    def test_missing_metadata_fields_handled(self) -> None:
        sparse_hit = [{"id": "x", "content": "text", "score": 0.8, "metadata": {}}]
        store = FakeVectorStore(hits=sparse_hit)
        retriever = SemanticRetriever(store=store)
        results = retriever.search("query")
        assert results[0].citation.source == "unknown"
        assert results[0].citation.chunk_index is None


# ── Chroma where-clause builder tests ──────────────────────────────────


class TestBuildChromaWhere:
    @pytest.fixture(autouse=True)
    def _skip_if_chroma_broken(self) -> None:
        """Skip if chromadb can't be imported (pydantic v1/v2 conflict)."""
        try:
            from agentic_rag.retrieval.chroma_store import _build_chroma_where  # noqa: F401
        except Exception:
            pytest.skip("chromadb not importable in this environment")

    def test_single_filter(self) -> None:
        from agentic_rag.retrieval.chroma_store import _build_chroma_where

        where = _build_chroma_where([MetadataFilter.equals("source", "a.md")])
        assert where == {"source": {"$eq": "a.md"}}

    def test_multiple_filters_produce_and(self) -> None:
        from agentic_rag.retrieval.chroma_store import _build_chroma_where

        filters = [
            MetadataFilter.equals("source", "a.md"),
            MetadataFilter(field="page", operator="gte", value=5),
        ]
        where = _build_chroma_where(filters)
        assert "$and" in where
        assert len(where["$and"]) == 2

    def test_none_when_empty(self) -> None:
        from agentic_rag.retrieval.chroma_store import _build_chroma_where

        assert _build_chroma_where([]) is None

    def test_unsupported_operator_raises(self) -> None:
        from agentic_rag.retrieval.chroma_store import _build_chroma_where

        with pytest.raises(ValueError, match="Unsupported filter operator"):
            _build_chroma_where([MetadataFilter(field="x", operator="regex", value=".*")])


# ── LangChain adapter tests ───────────────────────────────────────────


class TestLangChainAdapter:
    def test_adapter_returns_documents(self, retriever: SemanticRetriever) -> None:
        lc_retriever = retriever.as_langchain_retriever(k=2)
        docs = lc_retriever.invoke("Kubeflow")
        assert len(docs) == 2
        assert docs[0].page_content == SAMPLE_HITS[0]["content"]

    def test_adapter_embeds_citation_in_metadata(self, retriever: SemanticRetriever) -> None:
        lc_retriever = retriever.as_langchain_retriever(k=1)
        docs = lc_retriever.invoke("anything")
        meta = docs[0].metadata
        assert "_citation" in meta
        assert meta["_citation"]["source"] == "kubeflow_guide.md"
