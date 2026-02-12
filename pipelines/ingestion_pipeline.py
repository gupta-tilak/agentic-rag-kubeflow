"""KFP v2 pipeline — Modular RAG ingestion workflow.

This pipeline decomposes document ingestion into four discrete, reusable
stages connected by KFP Dataset artifacts:

    fetch → chunk → embed → index

Each stage is a standalone ``@dsl.component`` that can be tested locally,
swapped for a different implementation, or reused in other pipelines.

Compile
-------
    python -m pipelines.ingestion_pipeline --compile

Run locally (requires ``kfp`` installed)
-----------------------------------------
    python -m pipelines.ingestion_pipeline --compile
    # then submit the YAML to a KFP-compatible backend
"""

from kfp import compiler, dsl

from pipelines.components.chunk import chunk_text
from pipelines.components.embed import generate_embeddings
from pipelines.components.evaluate import evaluate_retrieval
from pipelines.components.fetch import fetch_documents
from pipelines.components.store import store_vectors


# ──────────────────────────────────────────────────────────────────────
# Pipeline definition
# ──────────────────────────────────────────────────────────────────────


@dsl.pipeline(
    name="rag-ingestion-pipeline",
    description=(
        "Modular RAG ingestion: fetch documents → chunk text → "
        "generate embeddings → index vectors.  Optionally evaluates "
        "retrieval quality after indexing."
    ),
)
def ingestion_pipeline(
    # ── Source ──────────────────────────────────────────────────────
    urls: str = '["/data/documents"]',
    source_type: str = "directory",
    glob_pattern: str = "**/*.md",
    request_headers: str = "{}",
    request_timeout: int = 60,
    max_retries: int = 3,
    # ── Chunking ───────────────────────────────────────────────────
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: str = '["\\n\\n", "\\n", ". ", " ", ""]',
    # ── Embedding ──────────────────────────────────────────────────
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    embed_batch_size: int = 64,
    normalize_embeddings: bool = True,
    # ── Vector DB ──────────────────────────────────────────────────
    chroma_host: str = "chroma.kubeflow.svc.cluster.local",
    chroma_port: int = 8000,
    collection_name: str = "agentic_rag",
    vector_db_type: str = "chroma",
    distance_metric: str = "cosine",
    upsert_batch_size: int = 5000,
    # ── Evaluation (optional) ──────────────────────────────────────
    run_evaluation: bool = False,
    eval_queries_path: str = "/data/eval_queries.json",
    retrieval_k: int = 5,
) -> None:
    """Four-step ingestion: fetch → chunk → embed → index.

    All inter-step data flows through KFP ``Dataset`` artifacts so that
    each component can be independently developed, tested, and versioned.
    Every component also emits a ``Metrics`` artifact for observability.

    Parameters
    ----------
    urls:
        JSON list of source URLs, directory paths, or GCS URIs.
    source_type:
        ``"url"`` | ``"directory"`` | ``"gcs"``
    glob_pattern:
        File-matching glob (directory mode only).
    request_headers:
        JSON-encoded HTTP headers (url mode only).
    request_timeout:
        Per-request timeout in seconds.
    max_retries:
        Retry attempts for transient HTTP errors.
    chunk_size / chunk_overlap:
        Text chunking parameters.
    separators:
        JSON-encoded list of split separators.
    embedding_model:
        HuggingFace model identifier for embedding.
    embed_batch_size:
        Forward-pass batch size for the embedding model.
    normalize_embeddings:
        Whether to L2-normalise embedding vectors.
    chroma_host / chroma_port / collection_name:
        Chroma connection details.
    vector_db_type:
        Vector-store backend (currently ``"chroma"``).
    distance_metric:
        ``"cosine"`` | ``"l2"`` | ``"ip"``
    upsert_batch_size:
        Max records per upsert call.
    run_evaluation:
        Whether to run retrieval evaluation after indexing.
    eval_queries_path:
        Path to the evaluation query set (JSON).
    retrieval_k:
        Number of documents to retrieve per eval query.
    """
    # Step 1 — Fetch (accepts list of URLs / paths)
    fetch_task = fetch_documents(
        urls=urls,
        source_type=source_type,
        glob_pattern=glob_pattern,
        request_headers=request_headers,
        request_timeout=request_timeout,
        max_retries=max_retries,
    )

    # Step 2 — Chunk with overlap
    chunk_task = chunk_text(
        raw_documents=fetch_task.outputs["raw_documents"],
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=separators,
    )

    # Step 3 — Generate embeddings
    embed_task = generate_embeddings(
        chunked_documents=chunk_task.outputs["chunked_documents"],
        embedding_model=embedding_model,
        batch_size=embed_batch_size,
        normalize_embeddings=normalize_embeddings,
    )

    # Step 4 — Index in vector DB
    store_task = store_vectors(
        embedded_documents=embed_task.outputs["embedded_documents"],
        chroma_host=chroma_host,
        chroma_port=chroma_port,
        collection_name=collection_name,
        vector_db_type=vector_db_type,
        distance_metric=distance_metric,
        upsert_batch_size=upsert_batch_size,
    )

    # Optional — Evaluate retrieval quality
    with dsl.If(run_evaluation == True):  # noqa: E712
        evaluate_retrieval(
            eval_queries_path=eval_queries_path,
            chroma_host=chroma_host,
            chroma_port=chroma_port,
            collection_name=collection_name,
            embedding_model=embedding_model,
            k=retrieval_k,
        ).after(store_task)


# ──────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG ingestion pipeline")
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile pipeline to YAML",
    )
    parser.add_argument(
        "--output",
        default="pipelines/compiled/ingestion_pipeline.yaml",
        help="Output path for compiled YAML",
    )
    args = parser.parse_args()

    if args.compile:
        compiler.Compiler().compile(ingestion_pipeline, args.output)
        print(f"Pipeline compiled → {args.output}")
