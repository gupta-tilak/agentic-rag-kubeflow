"""Text chunking strategies."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_text_splitters import RecursiveCharacterTextSplitter

if TYPE_CHECKING:
    from langchain_core.documents import Document


def chunk_documents(
    documents: list[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> list[Document]:
    """Split *documents* into smaller chunks for embedding.

    Parameters
    ----------
    documents:
        Source documents produced by a loader.
    chunk_size:
        Maximum number of characters per chunk.
    chunk_overlap:
        Number of overlapping characters between consecutive chunks.

    Returns
    -------
    list[Document]
        Chunked documents ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    return splitter.split_documents(documents)
