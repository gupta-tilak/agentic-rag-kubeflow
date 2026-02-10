"""KFP v2 component — Evaluate retrieval quality."""

from kfp import dsl


@dsl.component(
    base_image="python:3.11-slim",
    packages_to_install=[
        "langchain",
        "langchain-community",
        "langchain-huggingface",
        "chromadb",
        "sentence-transformers",
    ],
)
def evaluate_retrieval(
    eval_queries_path: str,
    chroma_host: str,
    chroma_port: int,
    collection_name: str,
    embedding_model: str,
    k: int = 5,
) -> float:
    """Run a simple hit-rate evaluation over a set of test queries.

    Parameters
    ----------
    eval_queries_path:
        Path to a JSON file with ``[{"query": "...", "expected_source": "..."}]``.
    chroma_host / chroma_port / collection_name:
        Chroma connection parameters.
    embedding_model:
        HuggingFace model identifier.
    k:
        Number of documents to retrieve per query.

    Returns
    -------
    float
        Hit rate ∈ [0, 1].
    """
    import json

    import chromadb
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    with open(eval_queries_path) as f:
        eval_set = json.load(f)

    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    vs = Chroma(
        client=client,
        collection_name=collection_name,
        embedding_function=embeddings,
    )
    retriever = vs.as_retriever(search_kwargs={"k": k})

    hits = 0
    for item in eval_set:
        docs = retriever.invoke(item["query"])
        sources = [d.metadata.get("source", "") for d in docs]
        if item["expected_source"] in sources:
            hits += 1

    hit_rate = hits / len(eval_set) if eval_set else 0.0
    print(f"Hit rate: {hit_rate:.2%} ({hits}/{len(eval_set)})")
    return hit_rate
