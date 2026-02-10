"""Document loaders — thin wrappers around LangChain document loaders."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)

if TYPE_CHECKING:
    from langchain_core.documents import Document


def load_directory(path: str | Path, glob: str = "**/*.*") -> list[Document]:
    """Recursively load all supported documents from *path*.

    Parameters
    ----------
    path:
        Root directory containing source documents.
    glob:
        File-matching pattern forwarded to ``DirectoryLoader``.

    Returns
    -------
    list[Document]
        Flat list of LangChain ``Document`` objects with metadata.
    """
    loader = DirectoryLoader(
        str(path),
        glob=glob,
        loader_cls=TextLoader,  # type: ignore[arg-type]
        show_progress=True,
        use_multithreading=True,
    )
    return loader.load()


def load_pdf(path: str | Path) -> list[Document]:
    """Load a single PDF file."""
    return PyPDFLoader(str(path)).load()


def load_markdown(path: str | Path) -> list[Document]:
    """Load a single Markdown file."""
    return UnstructuredMarkdownLoader(str(path)).load()
