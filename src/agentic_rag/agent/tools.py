"""LangGraph tool definitions exposed to the agent."""

from __future__ import annotations

from langchain_core.tools import tool

from agentic_rag.retrieval.retriever import get_retriever


@tool
def vector_search(query: str) -> str:
    """Search the knowledge base and return relevant passages.

    Parameters
    ----------
    query:
        Natural-language search query.

    Returns
    -------
    str
        Concatenated top-k document excerpts.
    """
    retriever = get_retriever(k=5)
    docs = retriever.invoke(query)
    return "\n\n---\n\n".join(doc.page_content for doc in docs)
