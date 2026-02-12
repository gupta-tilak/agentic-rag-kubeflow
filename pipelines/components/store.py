"""KFP v2 component — Index embedded documents in a vector database.

Step 4 of the RAG ingestion pipeline.  Reads the embedded JSON-Lines
Dataset and upserts vectors + metadata into the target vector store.

The component uses **deterministic IDs** (``chunk_id``) so re-runs are
idempotent — identical content is overwritten, not duplicated.

Supported backends:
    * ``chroma``  — via ``chromadb.HttpClient``
    * Extend for Pinecone / Weaviate / Qdrant by adding an ``elif`` branch.

Local testing
-------------
    from pipelines.components.store import store_vectors
    store_vectors.python_func(
        embedded_documents=_FakeArtifact("/tmp/embedded.jsonl"),
        chroma_host="localhost",
        chroma_port=8000,
        collection_name="test",
    )
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "chromadb>=0.5,<1",
    ],
)
def store_vectors(
    embedded_documents: dsl.Input[dsl.Dataset],
    chroma_host: str,
    chroma_port: int,
    collection_name: str,
    metrics: dsl.Output[dsl.Metrics],
    vector_db_type: str = "chroma",
    distance_metric: str = "cosine",
    upsert_batch_size: int = 5000,
) -> str:
    """Upsert pre-computed vectors into a vector database.

    Parameters
    ----------
    embedded_documents:
        Input Dataset — JSON-Lines produced by ``generate_embeddings``
        with at minimum ``chunk_id``, ``text``, ``embedding``, and
        arbitrarily rich metadata fields.
    chroma_host / chroma_port:
        Vector-store connection details.
    collection_name:
        Target collection / index name.
    metrics:
        Output Metrics artifact with indexing statistics.
    vector_db_type:
        Backend type (``"chroma"``).  Extend for others.
    distance_metric:
        Distance function (``cosine`` | ``l2`` | ``ip``).
    upsert_batch_size:
        Max records per upsert call (Chroma cap ≈ 41 666).

    Returns
    -------
    str
        Summary, e.g. ``"Indexed 256 vectors → collection 'agentic_rag'"``.
    """
    import hashlib
    import json
    import logging
    import time
    from pathlib import Path

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("store_vectors")

    # ── read embedded records ─────────────────────────────────────
    records: list[dict] = []
    with open(embedded_documents.path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d: %s", lineno, exc)

    if not records:
        metrics.log_metric("vectors_indexed", 0)
        return "No records to index."

    log.info("Read %d embedded records", len(records))

    # ── validate required keys ────────────────────────────────────
    required_keys = {"text", "embedding"}
    for i, rec in enumerate(records):
        missing = required_keys - rec.keys()
        if missing:
            raise ValueError(
                f"Record {i} missing required keys: {missing}"
            )

    # ── upsert ────────────────────────────────────────────────────
    if vector_db_type != "chroma":
        raise ValueError(
            f"Unsupported vector_db_type={vector_db_type!r}. "
            "Currently only 'chroma' is implemented."
        )

    import chromadb

    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": distance_metric},
    )

    # Prepare batch arrays
    ids: list[str] = []
    embeddings: list[list[float]] = []
    documents: list[str] = []
    metadatas: list[dict] = []

    for rec in records:
        # Use chunk_id if available, else hash the content
        cid = rec.get("chunk_id") or hashlib.sha256(
            rec["text"].encode()
        ).hexdigest()[:16]
        ids.append(cid)
        embeddings.append(rec["embedding"])
        documents.append(rec["text"])

        # Chroma metadata values must be flat str/int/float/bool
        meta = {}
        for k, v in rec.items():
            if k in ("text", "embedding"):
                continue
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
        metadatas.append(meta)

    t0 = time.monotonic()
    batches = 0
    for start in range(0, len(ids), upsert_batch_size):
        end = start + upsert_batch_size
        collection.upsert(
            ids=ids[start:end],
            embeddings=embeddings[start:end],
            documents=documents[start:end],
            metadatas=metadatas[start:end],
        )
        batches += 1
        log.info("  upserted batch %d (%d-%d)", batches, start, min(end, len(ids)))
    elapsed = time.monotonic() - t0

    log.info("Indexed %d vectors in %.1fs (%d batches)",
             len(ids), elapsed, batches)

    # KFP Metrics
    metrics.log_metric("vectors_indexed", len(ids))
    metrics.log_metric("upsert_batches", batches)
    metrics.log_metric("index_elapsed_seconds", round(elapsed, 2))
    metrics.log_metric("collection_name", collection_name)

    msg = (f"Indexed {len(ids)} vectors → collection '{collection_name}' "
           f"in {elapsed:.1f}s")
    log.info(msg)
    return msg
