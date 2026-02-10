"""KFP v2 component — Ingest documents into the vector store."""

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
def ingest_documents(
    source_path: str,
    chroma_host: str,
    chroma_port: int,
    collection_name: str,
    embedding_model: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> str:
    """Load, chunk, embed, and store documents.

    Parameters
    ----------
    source_path:
        Path or URI to the directory of source documents.
    chroma_host / chroma_port:
        Chroma vector-store connection details.
    collection_name:
        Target Chroma collection.
    embedding_model:
        HuggingFace model identifier for embedding.
    chunk_size / chunk_overlap:
        Chunking parameters.

    Returns
    -------
    str
        Summary message with the number of chunks stored.
    """
    import chromadb
    from langchain_community.document_loaders import DirectoryLoader, TextLoader
    from langchain_community.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    # Load
    loader = DirectoryLoader(source_path, loader_cls=TextLoader)
    docs = loader.load()

    # Chunk
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = splitter.split_documents(docs)

    # Embed + Store
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
    client = chromadb.HttpClient(host=chroma_host, port=chroma_port)
    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        client=client,
        collection_name=collection_name,
    )

    msg = f"Ingested {len(chunks)} chunks from {len(docs)} documents."
    print(msg)
    return msg
