"""KFP v2 component — Chunk normalised documents with overlap.

Step 2 of the RAG ingestion pipeline.  Reads the structured JSON-Lines
Dataset produced by ``fetch_documents`` and splits each document's text
into overlapping chunks suitable for embedding.

Structured output contract (one JSON object per line)::

    {
      "chunk_id":     "<doc_id>_<chunk_index>",
      "doc_id":       "<parent document hash>",
      "source":       "<original URL or file path>",
      "title":        "<inherited from parent>",
      "text":         "<chunk text>",
      "chunk_index":  0,
      "chunk_count":  12,
      "char_count":   487,
      "token_estimate": 122
    }

Local testing
-------------
    from pipelines.components.chunk import chunk_text
    chunk_text.python_func(
        raw_documents=_FakeArtifact("/tmp/raw.jsonl"),
        chunked_documents=_FakeArtifact("/tmp/chunked.jsonl"),
    )
"""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "langchain>=0.2,<1",
        "langchain-text-splitters>=0.2,<1",
    ],
)
def chunk_text(
    raw_documents: dsl.Input[dsl.Dataset],
    chunked_documents: dsl.Output[dsl.Dataset],
    metrics: dsl.Output[dsl.Metrics],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    separators: str = '["\\n\\n", "\\n", ". ", " ", ""]',
) -> str:
    """Split normalised documents into overlapping text chunks.

    Parameters
    ----------
    raw_documents:
        Input Dataset — JSON-Lines produced by ``fetch_documents`` with
        at minimum ``doc_id``, ``source``, ``title``, and ``text`` keys.
    chunked_documents:
        Output Dataset — JSON-Lines, one record per chunk (see module docstring).
    metrics:
        Output Metrics artifact with chunking statistics.
    chunk_size:
        Maximum character length of each chunk.
    chunk_overlap:
        Number of overlapping characters between consecutive chunks.
    separators:
        JSON-encoded list of split boundaries, in priority order.

    Returns
    -------
    str
        Summary, e.g. ``"Produced 256 chunks from 42 documents"``.
    """
    import json
    import logging
    from pathlib import Path

    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger("chunk_text")

    # ── validate params ───────────────────────────────────────────
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be < chunk_size ({chunk_size})"
        )

    # ── read raw documents ────────────────────────────────────────
    raw_records: list[dict] = []
    with open(raw_documents.path) as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                log.warning("Skipping malformed line %d: %s", lineno, exc)
                continue
            if "text" not in obj:
                log.warning("Skipping line %d: missing 'text' key", lineno)
                continue
            raw_records.append(obj)

    log.info("Read %d documents from input artifact", len(raw_records))

    # ── configure splitter ────────────────────────────────────────
    sep_list = json.loads(separators) if isinstance(separators, str) else separators
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=sep_list,
    )

    # ── chunk each document ───────────────────────────────────────
    all_chunks: list[dict] = []
    for rec in raw_records:
        doc_id = rec.get("doc_id", "unknown")
        source = rec.get("source", "")
        title = rec.get("title", "")
        text = rec["text"]

        lc_docs = splitter.split_documents(
            [Document(page_content=text, metadata={"source": source})]
        )

        chunk_count = len(lc_docs)
        for idx, chunk in enumerate(lc_docs):
            chunk_text_content = chunk.page_content
            all_chunks.append({
                "chunk_id": f"{doc_id}_{idx}",
                "doc_id": doc_id,
                "source": source,
                "title": title,
                "text": chunk_text_content,
                "chunk_index": idx,
                "chunk_count": chunk_count,
                "char_count": len(chunk_text_content),
                "token_estimate": len(chunk_text_content) // 4,  # rough ≈4 chars/token
            })

    log.info("Produced %d chunks from %d documents", len(all_chunks), len(raw_records))

    # ── write output ──────────────────────────────────────────────
    out_path = Path(chunked_documents.path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        for chunk in all_chunks:
            fh.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    # artifact metadata
    chunked_documents.metadata["num_chunks"] = len(all_chunks)
    chunked_documents.metadata["num_documents"] = len(raw_records)
    chunked_documents.metadata["chunk_size"] = chunk_size
    chunked_documents.metadata["chunk_overlap"] = chunk_overlap
    total_chars = sum(c["char_count"] for c in all_chunks)
    chunked_documents.metadata["total_chars"] = total_chars

    # KFP Metrics
    metrics.log_metric("chunks_produced", len(all_chunks))
    metrics.log_metric("documents_processed", len(raw_records))
    metrics.log_metric("avg_chunk_chars",
                       total_chars / len(all_chunks) if all_chunks else 0)

    msg = f"Produced {len(all_chunks)} chunks from {len(raw_records)} documents"
    log.info(msg)
    return msg
