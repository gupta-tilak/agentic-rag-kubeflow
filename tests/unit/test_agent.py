"""Unit tests for agent state and prompt construction."""

from langchain_core.documents import Document

from agentic_rag.agent.prompts import build_rag_prompt


def test_build_rag_prompt_returns_two_messages() -> None:
    """Prompt should contain a system message and a human message."""
    docs = [Document(page_content="Kubeflow is an ML platform.")]
    messages = build_rag_prompt("What is Kubeflow?", docs)
    assert len(messages) == 2


def test_build_rag_prompt_includes_context() -> None:
    """The human message should contain the document content."""
    content = "LangGraph enables agentic workflows."
    docs = [Document(page_content=content)]
    messages = build_rag_prompt("Tell me about LangGraph", docs)
    human_msg = messages[-1].content
    assert content in human_msg


def test_build_rag_prompt_includes_query() -> None:
    """The human message should contain the user's question."""
    query = "How does retrieval work?"
    docs = [Document(page_content="Retrieval uses vector search.")]
    messages = build_rag_prompt(query, docs)
    assert query in messages[-1].content
