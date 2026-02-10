"""Graph nodes — each function corresponds to one step in the agent graph."""

from __future__ import annotations

from langchain_core.messages import AIMessage

from agentic_rag.agent.state import AgentState
from agentic_rag.retrieval.retriever import get_retriever


def retrieve(state: AgentState) -> AgentState:
    """Retrieve relevant documents for the current query."""
    retriever = get_retriever(k=5)
    docs = retriever.invoke(state["query"])
    return {**state, "documents": docs}


def generate(state: AgentState) -> AgentState:
    """Generate an answer using the retrieved documents and LLM.

    This node builds a prompt from the retrieved context and delegates
    to the configured LLM.  The implementation is intentionally kept
    thin so it can be swapped for any LangChain-compatible chat model.
    """
    from agentic_rag.agent.prompts import build_rag_prompt
    from agentic_rag.agent.llm import get_llm

    llm = get_llm()
    prompt = build_rag_prompt(state["query"], state["documents"])
    response = llm.invoke(prompt)

    return {
        **state,
        "messages": [AIMessage(content=response.content)],
    }


def grade_documents(state: AgentState) -> str:
    """Conditional edge: decide whether retrieved docs are relevant.

    Returns
    -------
    str
        ``"generate"`` if documents are relevant, ``"retrieve"`` to retry.
    """
    if not state.get("documents"):
        return "retrieve"
    # Placeholder: always trust the retriever for now.
    return "generate"
