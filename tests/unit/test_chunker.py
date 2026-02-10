"""Unit tests for the chunker module."""

from langchain_core.documents import Document

from agentic_rag.ingestion.chunker import chunk_documents


def test_chunk_documents_splits_long_text() -> None:
    """A document longer than chunk_size should be split."""
    long_text = "word " * 500  # ~2500 chars
    docs = [Document(page_content=long_text, metadata={"source": "test"})]
    chunks = chunk_documents(docs, chunk_size=256, chunk_overlap=32)
    assert len(chunks) > 1


def test_chunk_documents_preserves_metadata() -> None:
    """Metadata from the source document should be preserved in chunks."""
    docs = [Document(page_content="Short text.", metadata={"source": "test.md"})]
    chunks = chunk_documents(docs, chunk_size=256, chunk_overlap=0)
    assert all(c.metadata.get("source") == "test.md" for c in chunks)


def test_chunk_documents_empty_input() -> None:
    """An empty list should return an empty list."""
    assert chunk_documents([]) == []
