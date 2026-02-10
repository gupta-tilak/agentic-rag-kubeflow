"""Prompt templates for the RAG agent."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

if TYPE_CHECKING:
    from langchain_core.documents import Document
    from langchain_core.messages import BaseMessage

SYSTEM_PROMPT = """\
You are a helpful assistant that answers questions using the provided context.
If the context does not contain enough information, say so honestly.
Always cite which part of the context supports your answer.
"""


def build_rag_prompt(query: str, documents: list[Document]) -> list[BaseMessage]:
    """Assemble the prompt messages for a retrieval-augmented generation call.

    Parameters
    ----------
    query:
        The user question.
    documents:
        Retrieved context chunks.

    Returns
    -------
    list[BaseMessage]
        A list of LangChain message objects ready for ``.invoke()``.
    """
    context = "\n\n---\n\n".join(doc.page_content for doc in documents)
    user_msg = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Provide a detailed answer based on the context above."
    )
    return [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_msg),
    ]
