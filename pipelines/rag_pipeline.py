"""KFP v2 pipeline — end-to-end RAG ingestion + evaluation."""

from __future__ import annotations

from kfp import compiler, dsl

from pipelines.components.evaluate import evaluate_retrieval
from pipelines.components.ingest import ingest_documents


@dsl.pipeline(
    name="agentic-rag-pipeline",
    description="Ingest documents, embed them, and evaluate retrieval quality.",
)
def rag_pipeline(
    source_path: str = "/data/documents",
    eval_queries_path: str = "/data/eval_queries.json",
    chroma_host: str = "chroma.kubeflow.svc.cluster.local",
    chroma_port: int = 8000,
    collection_name: str = "agentic_rag",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    retrieval_k: int = 5,
) -> None:
    """Two-step pipeline: ingest → evaluate."""
    ingest_task = ingest_documents(
        source_path=source_path,
        chroma_host=chroma_host,
        chroma_port=chroma_port,
        collection_name=collection_name,
        embedding_model=embedding_model,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    evaluate_retrieval(
        eval_queries_path=eval_queries_path,
        chroma_host=chroma_host,
        chroma_port=chroma_port,
        collection_name=collection_name,
        embedding_model=embedding_model,
        k=retrieval_k,
    ).after(ingest_task)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--compile", action="store_true", help="Compile pipeline to YAML")
    args = parser.parse_args()

    if args.compile:
        compiler.Compiler().compile(rag_pipeline, "pipelines/compiled/rag_pipeline.yaml")
        print("Pipeline compiled → pipelines/compiled/rag_pipeline.yaml")
