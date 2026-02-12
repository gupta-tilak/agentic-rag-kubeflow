"""Unit tests for the modular KFP ingestion components.

Each test exercises the *Python function* behind the ``@dsl.component``
decorator (``component.python_func``), so no Kubeflow cluster is needed.

Structured JSON contract flowing between components:

  fetch  → {doc_id, source, content_type, title, text, fetched_at, char_count}
  chunk  → {chunk_id, doc_id, source, title, text, chunk_index, chunk_count,
             char_count, token_estimate}
  embed  → chunk record + {embedding, embedding_model, embedding_dim}
  index  → reads embed output, upserts to vector DB
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ──────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────


class _FakeArtifact:
    """Minimal stand-in for ``dsl.Dataset`` / ``dsl.Metrics``."""

    def __init__(self, path: str) -> None:
        self.path = path
        self.metadata: dict = {}
        self._metrics: dict = {}

    def log_metric(self, name: str, value) -> None:
        self._metrics[name] = value


def _write_jsonl(path: str, records: list[dict]) -> None:
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")


def _read_jsonl(path: str) -> list[dict]:
    with open(path) as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ──────────────────────────────────────────────────────────────────────
# fetch_documents
# ──────────────────────────────────────────────────────────────────────


class TestFetchDocuments:
    """Tests for ``pipelines.components.fetch.fetch_documents``."""

    def test_fetch_from_directory(self, tmp_path: Path) -> None:
        """Loading from a local directory emits structured JSON records."""
        (tmp_path / "a.md").write_text("# Hello\nWorld")
        (tmp_path / "b.md").write_text("# Goodbye\nMoon")
        out_path = str(tmp_path / "output.jsonl")
        artifact = _FakeArtifact(out_path)
        metrics = _FakeArtifact(str(tmp_path / "metrics"))

        from pipelines.components.fetch import fetch_documents

        result = fetch_documents.python_func(
            urls=json.dumps([str(tmp_path)]),
            source_type="directory",
            raw_documents=artifact,
            metrics=metrics,
            glob_pattern="**/*.md",
        )

        records = _read_jsonl(out_path)
        assert len(records) == 2
        # Verify structured contract keys
        for rec in records:
            assert "doc_id" in rec
            assert "source" in rec
            assert "content_type" in rec
            assert "title" in rec
            assert "text" in rec
            assert "fetched_at" in rec
            assert "char_count" in rec
            assert rec["char_count"] == len(rec["text"])
        assert "Fetched 2" in result
        assert artifact.metadata["num_documents"] == 2
        assert metrics._metrics["documents_fetched"] == 2

    def test_fetch_from_multiple_urls(self, tmp_path: Path) -> None:
        """Passing multiple URLs fetches all of them."""
        artifact = _FakeArtifact(str(tmp_path / "out.jsonl"))
        metrics = _FakeArtifact(str(tmp_path / "metrics"))

        html_a = "<html><head><title>Page A</title></head><body><p>Alpha</p></body></html>"
        html_b = "<html><head><title>Page B</title></head><body><p>Beta</p></body></html>"

        from pipelines.components.fetch import fetch_documents

        responses = {
            "https://a.example.com": MagicMock(
                text=html_a,
                headers={"content-type": "text/html"},
                raise_for_status=MagicMock(),
            ),
            "https://b.example.com": MagicMock(
                text=html_b,
                headers={"content-type": "text/html"},
                raise_for_status=MagicMock(),
            ),
        }

        def mock_get(url, **kwargs):
            return responses[url]

        with patch("requests.get", side_effect=mock_get):
            result = fetch_documents.python_func(
                urls='["https://a.example.com", "https://b.example.com"]',
                source_type="url",
                raw_documents=artifact,
                metrics=metrics,
            )

        records = _read_jsonl(artifact.path)
        assert len(records) == 2
        assert records[0]["title"] == "Page A"
        assert records[1]["title"] == "Page B"
        assert "Alpha" in records[0]["text"]
        assert "Beta" in records[1]["text"]

    def test_fetch_normalises_text(self, tmp_path: Path) -> None:
        """Fetched text should have collapsed whitespace and no control chars."""
        artifact = _FakeArtifact(str(tmp_path / "out.jsonl"))
        metrics = _FakeArtifact(str(tmp_path / "metrics"))

        html = "<html><body><p>Hello   \t  world\x00\x01</p></body></html>"
        from pipelines.components.fetch import fetch_documents

        mock_resp = MagicMock(
            text=html,
            headers={"content-type": "text/html"},
            raise_for_status=MagicMock(),
        )
        with patch("requests.get", return_value=mock_resp):
            fetch_documents.python_func(
                urls='["https://example.com"]',
                source_type="url",
                raw_documents=artifact,
                metrics=metrics,
            )

        records = _read_jsonl(artifact.path)
        text = records[0]["text"]
        assert "   " not in text       # no runs of 3+ spaces
        assert "\x00" not in text      # no null bytes
        assert "\t" not in text        # tabs collapsed

    def test_fetch_unsupported_source_type(self, tmp_path: Path) -> None:
        """An unknown source type should raise ``ValueError``."""
        artifact = _FakeArtifact(str(tmp_path / "out.jsonl"))
        metrics = _FakeArtifact(str(tmp_path / "metrics"))
        from pipelines.components.fetch import fetch_documents

        with pytest.raises(ValueError, match="Unsupported source_type"):
            fetch_documents.python_func(
                urls='["something"]',
                source_type="ftp",
                raw_documents=artifact,
                metrics=metrics,
            )

    def test_fetch_invalid_urls_param(self, tmp_path: Path) -> None:
        """A non-list should raise ``ValueError``."""
        artifact = _FakeArtifact(str(tmp_path / "out.jsonl"))
        metrics = _FakeArtifact(str(tmp_path / "metrics"))
        from pipelines.components.fetch import fetch_documents

        with pytest.raises(ValueError, match="non-empty JSON list"):
            fetch_documents.python_func(
                urls='"not-a-list"',
                source_type="url",
                raw_documents=artifact,
                metrics=metrics,
            )

    def test_fetch_retries_on_failure(self, tmp_path: Path) -> None:
        """Transient HTTP errors should be retried."""
        import requests

        artifact = _FakeArtifact(str(tmp_path / "out.jsonl"))
        metrics = _FakeArtifact(str(tmp_path / "metrics"))
        from pipelines.components.fetch import fetch_documents

        fail_resp = MagicMock()
        fail_resp.raise_for_status.side_effect = requests.HTTPError("503")

        ok_resp = MagicMock(
            text="<html><body>OK</body></html>",
            headers={"content-type": "text/html"},
            raise_for_status=MagicMock(),
        )

        with patch("requests.get", side_effect=[fail_resp, ok_resp]):
            with patch("time.sleep"):   # don't actually wait
                result = fetch_documents.python_func(
                    urls='["https://flaky.example.com"]',
                    source_type="url",
                    raw_documents=artifact,
                    metrics=metrics,
                    max_retries=3,
                )

        assert "Fetched 1" in result


# ──────────────────────────────────────────────────────────────────────
# chunk_text
# ──────────────────────────────────────────────────────────────────────


class TestChunkText:
    """Tests for ``pipelines.components.chunk.chunk_text``."""

    @staticmethod
    def _make_fetch_record(text: str, source: str = "test.md") -> dict:
        """Create a record matching the fetch output contract."""
        import hashlib
        return {
            "doc_id": hashlib.sha256(text.encode()).hexdigest()[:16],
            "source": source,
            "content_type": "text/markdown",
            "title": "Test",
            "text": text,
            "fetched_at": "2026-02-12T00:00:00+00:00",
            "char_count": len(text),
        }

    def test_chunks_long_document(self, tmp_path: Path) -> None:
        in_path = str(tmp_path / "raw.jsonl")
        out_path = str(tmp_path / "chunked.jsonl")
        metrics = _FakeArtifact(str(tmp_path / "metrics"))

        _write_jsonl(in_path, [self._make_fetch_record("word " * 500)])

        from pipelines.components.chunk import chunk_text

        out_art = _FakeArtifact(out_path)
        chunk_text.python_func(
            raw_documents=_FakeArtifact(in_path),
            chunked_documents=out_art,
            metrics=metrics,
            chunk_size=256,
            chunk_overlap=32,
        )

        chunks = _read_jsonl(out_path)
        assert len(chunks) > 1
        # Verify structured contract
        for c in chunks:
            assert "chunk_id" in c
            assert "doc_id" in c
            assert "text" in c
            assert "chunk_index" in c
            assert "chunk_count" in c
            assert "char_count" in c
            assert "token_estimate" in c
            assert c["char_count"] == len(c["text"])

    def test_preserves_source_metadata(self, tmp_path: Path) -> None:
        in_path = str(tmp_path / "raw.jsonl")
        out_path = str(tmp_path / "chunked.jsonl")
        metrics = _FakeArtifact(str(tmp_path / "metrics"))

        _write_jsonl(in_path, [self._make_fetch_record("Short.", source="f.md")])

        from pipelines.components.chunk import chunk_text

        chunk_text.python_func(
            raw_documents=_FakeArtifact(in_path),
            chunked_documents=_FakeArtifact(out_path),
            metrics=metrics,
        )

        chunks = _read_jsonl(out_path)
        assert all(c["source"] == "f.md" for c in chunks)
        assert all(c["title"] == "Test" for c in chunks)

    def test_empty_input(self, tmp_path: Path) -> None:
        in_path = str(tmp_path / "raw.jsonl")
        out_path = str(tmp_path / "chunked.jsonl")
        metrics = _FakeArtifact(str(tmp_path / "metrics"))
        Path(in_path).write_text("")

        from pipelines.components.chunk import chunk_text

        out_art = _FakeArtifact(out_path)
        chunk_text.python_func(
            raw_documents=_FakeArtifact(in_path),
            chunked_documents=out_art,
            metrics=metrics,
        )

        assert _read_jsonl(out_path) == []
        assert out_art.metadata["num_chunks"] == 0

    def test_overlap_gte_chunk_size_raises(self, tmp_path: Path) -> None:
        in_path = str(tmp_path / "raw.jsonl")
        out_path = str(tmp_path / "chunked.jsonl")
        metrics = _FakeArtifact(str(tmp_path / "metrics"))
        _write_jsonl(in_path, [self._make_fetch_record("Hello")])

        from pipelines.components.chunk import chunk_text

        with pytest.raises(ValueError, match="chunk_overlap.*must be"):
            chunk_text.python_func(
                raw_documents=_FakeArtifact(in_path),
                chunked_documents=_FakeArtifact(out_path),
                metrics=metrics,
                chunk_size=100,
                chunk_overlap=100,
            )

    def test_chunk_id_uses_doc_id(self, tmp_path: Path) -> None:
        """chunk_id should be ``<doc_id>_<index>``."""
        in_path = str(tmp_path / "raw.jsonl")
        out_path = str(tmp_path / "chunked.jsonl")
        metrics = _FakeArtifact(str(tmp_path / "metrics"))
        rec = self._make_fetch_record("Short text.")
        _write_jsonl(in_path, [rec])

        from pipelines.components.chunk import chunk_text

        chunk_text.python_func(
            raw_documents=_FakeArtifact(in_path),
            chunked_documents=_FakeArtifact(out_path),
            metrics=metrics,
        )
        chunks = _read_jsonl(out_path)
        assert chunks[0]["chunk_id"] == f"{rec['doc_id']}_0"


# ──────────────────────────────────────────────────────────────────────
# generate_embeddings
# ──────────────────────────────────────────────────────────────────────


class TestGenerateEmbeddings:
    """Tests for ``pipelines.components.embed.generate_embeddings``."""

    @staticmethod
    def _make_chunk_record(text: str) -> dict:
        return {
            "chunk_id": f"abc123_{0}",
            "doc_id": "abc123",
            "source": "test.md",
            "title": "Test",
            "text": text,
            "chunk_index": 0,
            "chunk_count": 1,
            "char_count": len(text),
            "token_estimate": len(text) // 4,
        }

    def test_produces_embeddings(self, tmp_path: Path) -> None:
        in_path = str(tmp_path / "chunked.jsonl")
        out_path = str(tmp_path / "embedded.jsonl")
        metrics = _FakeArtifact(str(tmp_path / "metrics"))

        _write_jsonl(in_path, [
            self._make_chunk_record("Hello world"),
            self._make_chunk_record("Goodbye moon"),
        ])

        from pipelines.components.embed import generate_embeddings

        out_art = _FakeArtifact(out_path)
        generate_embeddings.python_func(
            chunked_documents=_FakeArtifact(in_path),
            embedded_documents=out_art,
            metrics=metrics,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            batch_size=2,
        )

        records = _read_jsonl(out_path)
        assert len(records) == 2
        for rec in records:
            assert isinstance(rec["embedding"], list)
            assert len(rec["embedding"]) > 0
            assert rec["embedding_model"] == "sentence-transformers/all-MiniLM-L6-v2"
            assert rec["embedding_dim"] > 0
        assert out_art.metadata["num_embedded"] == 2
        assert metrics._metrics["chunks_embedded"] == 2

    def test_empty_input(self, tmp_path: Path) -> None:
        in_path = str(tmp_path / "chunked.jsonl")
        out_path = str(tmp_path / "embedded.jsonl")
        metrics = _FakeArtifact(str(tmp_path / "metrics"))
        Path(in_path).write_text("")

        from pipelines.components.embed import generate_embeddings

        result = generate_embeddings.python_func(
            chunked_documents=_FakeArtifact(in_path),
            embedded_documents=_FakeArtifact(out_path),
            metrics=metrics,
        )
        assert result == "No chunks to embed."


# ──────────────────────────────────────────────────────────────────────
# store_vectors (index)
# ──────────────────────────────────────────────────────────────────────


class TestStoreVectors:
    """Tests for ``pipelines.components.store.store_vectors``."""

    def test_index_to_chroma(self, tmp_path: Path) -> None:
        in_path = str(tmp_path / "embedded.jsonl")
        metrics = _FakeArtifact(str(tmp_path / "metrics"))

        _write_jsonl(in_path, [
            {
                "chunk_id": "abc123_0",
                "doc_id": "abc123",
                "source": "a.md",
                "title": "A",
                "text": "Hello",
                "chunk_index": 0,
                "chunk_count": 1,
                "char_count": 5,
                "token_estimate": 1,
                "embedding": [0.1, 0.2, 0.3],
                "embedding_model": "test",
                "embedding_dim": 3,
            },
            {
                "chunk_id": "def456_0",
                "doc_id": "def456",
                "source": "b.md",
                "title": "B",
                "text": "World",
                "chunk_index": 0,
                "chunk_count": 1,
                "char_count": 5,
                "token_estimate": 1,
                "embedding": [0.4, 0.5, 0.6],
                "embedding_model": "test",
                "embedding_dim": 3,
            },
        ])

        from pipelines.components.store import store_vectors

        mock_collection = MagicMock()
        mock_client = MagicMock()
        mock_client.get_or_create_collection.return_value = mock_collection

        # Create a mock chromadb module to avoid Python 3.14 pydantic v1 issue
        mock_chromadb = MagicMock()
        mock_chromadb.HttpClient.return_value = mock_client

        with patch.dict("sys.modules", {"chromadb": mock_chromadb}):
            result = store_vectors.python_func(
                embedded_documents=_FakeArtifact(in_path),
                chroma_host="localhost",
                chroma_port=8000,
                collection_name="test_collection",
                metrics=metrics,
            )

        assert "Indexed 2 vectors" in result
        mock_collection.upsert.assert_called_once()
        call_kwargs = mock_collection.upsert.call_args
        assert len(call_kwargs.kwargs["ids"]) == 2
        # Verify chunk_id is used as the vector ID
        assert call_kwargs.kwargs["ids"] == ["abc123_0", "def456_0"]
        assert metrics._metrics["vectors_indexed"] == 2

    def test_index_empty_input(self, tmp_path: Path) -> None:
        in_path = str(tmp_path / "embedded.jsonl")
        metrics = _FakeArtifact(str(tmp_path / "metrics"))
        Path(in_path).write_text("")

        from pipelines.components.store import store_vectors

        result = store_vectors.python_func(
            embedded_documents=_FakeArtifact(in_path),
            chroma_host="localhost",
            chroma_port=8000,
            collection_name="test",
            metrics=metrics,
        )
        assert result == "No records to index."

    def test_index_unsupported_db(self, tmp_path: Path) -> None:
        in_path = str(tmp_path / "embedded.jsonl")
        metrics = _FakeArtifact(str(tmp_path / "metrics"))
        _write_jsonl(in_path, [{"text": "x", "embedding": [0.1]}])

        from pipelines.components.store import store_vectors

        with pytest.raises(ValueError, match="Unsupported vector_db_type"):
            store_vectors.python_func(
                embedded_documents=_FakeArtifact(in_path),
                chroma_host="localhost",
                chroma_port=8000,
                collection_name="c",
                metrics=metrics,
                vector_db_type="pinecone",
            )

    def test_index_missing_embedding_key_raises(self, tmp_path: Path) -> None:
        """Records without 'embedding' should fail validation."""
        in_path = str(tmp_path / "embedded.jsonl")
        metrics = _FakeArtifact(str(tmp_path / "metrics"))
        _write_jsonl(in_path, [{"text": "hello"}])  # no embedding

        from pipelines.components.store import store_vectors

        with pytest.raises(ValueError, match="missing required keys"):
            store_vectors.python_func(
                embedded_documents=_FakeArtifact(in_path),
                chroma_host="localhost",
                chroma_port=8000,
                collection_name="c",
                metrics=metrics,
            )
