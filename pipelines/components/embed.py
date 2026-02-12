"""KFP v2 component — Generate embeddings for chunked documents.

Step 3 of the RAG ingestion pipeline.  Reads the chunked JSON-Lines
Dataset, generates vector embeddings in batches, and writes an enriched
JSON-Lines Dataset for downstream indexing.

Structured output contract (one JSON object per line)::

    {
      "chunk_id":       "<doc_id>_<chunk_index>",
      "doc_id":         "<parent document hash>",
      "source":         "<original URL or file path>",
      "title":          "<inherited from parent>",
      "text":           "<chunk text>",
      "chunk_index":    0,
      "chunk_count":    12,
      "char_count":     487,
      "token_estimate": 122,
      "embedding":      [0.012, -0.034, ...],
      "embedding_model":"sentence-transformers/all-MiniLM-L6-v2",
      "embedding_dim":  384
    }

Local testing
-------------
    from pipelines.components.embed import generate_embeddings
    generate_embeddings.python_func(
        chunked_documents=_FakeArtifact("/tmp/chunked.jsonl"),
        embedded_documents=_FakeArtifact("/tmp/embedded.jsonl"),
    )
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "langchain>=0.2,<1",
        "langchain-huggingface>=0.1,<1",
        "sentence-transformers>=3,<4",
    ],
)
def generate_embeddings(
    chunked_documents: dsl.Input[dsl.Dataset],
    embedded_documents: dsl.Output[dsl.Dataset],
    metrics: dsl.Output[dsl.Metrics],
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 64,
    normalize_embeddings: bool = True,
) -> str:
    """Embed every text chunk and persist vectors alongside content.

    Parameters
    ----------
    chunked_documents:
        Input Dataset — JSON-Lines produced by ``chunk_text`` with at
        minimum ``chunk_id`` and ``text`` keys.
    embedded_documents:
        Output Dataset — JSON-Lines, each record enriched with
        ``embedding``, ``embedding_model``, and ``embedding_dim`` keys.
    metrics:
        Output Metrics artifact with embedding statistics.
    embedding_model:
        HuggingFace sentence-transformer model identifier.
    batch_size:
        Number of texts to embed per forward pass.
    normalize_embeddings:
        Whether to L2-normalise vectors (recommended for cosine similarity).

    Returns
    -------
    str
        Summary, e.g. ``"Embedded 256 chunks (dim=384)"``.
    """
    import json
    import logging
    import time
    from pathlib import Path

    from langchain_huggingface import HuggingFaceEmbeddings

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("generate_embeddings")

    # ── read chunks ───────────────────────────────────────────────
    records: list[dict] = []
    with open(chunked_documents.path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d: %s", lineno, exc)

    if not records:
        # Write empty file and return early
        out_path = Path(embedded_documents.path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("")
        embedded_documents.metadata["num_embedded"] = 0
        embedded_documents.metadata["embedding_dim"] = 0
        metrics.log_metric("chunks_embedded", 0)
        return "No chunks to embed."

    texts = [r["text"] for r in records]
    log.info("Embedding %d chunks with model=%s, batch_size=%d",
             len(texts), embedding_model, batch_size)

    # ── embed in batches ──────────────────────────────────────────
    model_kwargs = {}
    encode_kwargs = {"normalize_embeddings": normalize_embeddings}
    embedder = HuggingFaceEmbeddings(
        model_name=embedding_model,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

    all_embeddings: list[list[float]] = []
    t0 = time.monotonic()
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        all_embeddings.extend(embedder.embed_documents(batch))
        log.info("  embedded %d / %d", len(all_embeddings), len(texts))
    elapsed = time.monotonic() - t0

    dim = len(all_embeddings[0])
    log.info("Embedding complete: %d vectors (dim=%d) in %.1fs",
             len(all_embeddings), dim, elapsed)

    # ── write output ──────────────────────────────────────────────
    out_path = Path(embedded_documents.path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for rec, emb in zip(records, all_embeddings):
            rec["embedding"] = emb
            rec["embedding_model"] = embedding_model
            rec["embedding_dim"] = dim
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # artifact metadata
    embedded_documents.metadata["num_embedded"] = len(all_embeddings)
    embedded_documents.metadata["embedding_dim"] = dim
    embedded_documents.metadata["embedding_model"] = embedding_model
    embedded_documents.metadata["elapsed_seconds"] = round(elapsed, 2)

    # KFP Metrics
    metrics.log_metric("chunks_embedded", len(all_embeddings))
    metrics.log_metric("embedding_dim", dim)
    metrics.log_metric("embed_elapsed_seconds", round(elapsed, 2))
    metrics.log_metric("chunks_per_second",
                       round(len(all_embeddings) / elapsed, 1) if elapsed > 0 else 0)

    msg = f"Embedded {len(all_embeddings)} chunks (dim={dim}) in {elapsed:.1f}s"
    log.info(msg)
    return msg
