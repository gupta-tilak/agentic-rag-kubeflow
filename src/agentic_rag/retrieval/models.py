"""Domain models for retrieval results and citation tracking."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class MetadataFilter(BaseModel):
    """Declarative metadata filter for vector-store queries.

    Attributes
    ----------
    field:
        The metadata key to filter on (e.g. ``"source"``, ``"author"``).
    operator:
        Comparison operator — one of ``eq``, ``ne``, ``gt``, ``gte``,
        ``lt``, ``lte``, ``in``, ``nin``.
    value:
        The value (or list of values for ``in`` / ``nin``) to compare against.
    """

    field: str
    operator: str = "eq"
    value: Any = None

    # -- helpers for common filters ------------------------------------------

    @classmethod
    def equals(cls, field: str, value: Any) -> MetadataFilter:
        return cls(field=field, operator="eq", value=value)

    @classmethod
    def not_equals(cls, field: str, value: Any) -> MetadataFilter:
        return cls(field=field, operator="ne", value=value)

    @classmethod
    def one_of(cls, field: str, values: list[Any]) -> MetadataFilter:
        return cls(field=field, operator="in", value=values)


class Citation(BaseModel):
    """Provenance record linking a retrieved chunk back to its source document.

    Every retrieval result carries a ``Citation`` so that consuming layers
    (agents, UIs, evaluation harnesses) can trace *exactly* where the
    information came from.

    Attributes
    ----------
    citation_id:
        Unique identifier for this citation instance.
    document_id:
        The vector-store ID of the chunk (``None`` when unknown).
    source:
        Human-readable source locator — file path, URL, etc.
    chunk_index:
        Ordinal position of the chunk within the source document.
    page:
        Page number (if applicable, e.g. PDF sources).
    score:
        Similarity / relevance score returned by the vector store.
    metadata:
        Arbitrary extra metadata attached to the original document.
    retrieved_at:
        UTC timestamp of when the retrieval happened.
    """

    citation_id: str = Field(default_factory=lambda: uuid4().hex[:12])
    document_id: str | None = None
    source: str = "unknown"
    chunk_index: int | None = None
    page: int | None = None
    score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    def short_ref(self) -> str:
        """Return a compact ``[source§chunk]`` reference string."""
        chunk = self.chunk_index if self.chunk_index is not None else "?"
        return f"[{self.source}§{chunk}]"


class RetrievalResult(BaseModel):
    """A single retrieved passage together with its citation."""

    content: str
    citation: Citation

    def __str__(self) -> str:  # noqa: D105
        return f"{self.citation.short_ref()} {self.content[:120]}…"
