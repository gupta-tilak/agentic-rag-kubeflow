"""KFP v2 components — each file exports one @dsl.component."""

from pipelines.components.chunk import chunk_text
from pipelines.components.embed import generate_embeddings
from pipelines.components.evaluate import evaluate_retrieval
from pipelines.components.fetch import fetch_documents
from pipelines.components.ingest import ingest_documents
from pipelines.components.store import store_vectors

__all__ = [
    "chunk_text",
    "evaluate_retrieval",
    "fetch_documents",
    "generate_embeddings",
    "ingest_documents",
    "store_vectors",
]
