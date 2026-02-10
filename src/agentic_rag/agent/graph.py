"""LangGraph graph definition — the agentic RAG workflow."""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agentic_rag.agent.nodes import generate, grade_documents, retrieve
from agentic_rag.agent.state import AgentState


def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph agent.

    Graph topology::

        [START] ──► retrieve ──► grade_documents
                                   │
                       ┌───── relevant? ─────┐
                       ▼                     ▼
                   generate               retrieve  (retry)
                       │
                       ▼
                     [END]
    """
    workflow = StateGraph(AgentState)

    # -- Nodes ---------------------------------------------------------------
    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    # -- Edges ---------------------------------------------------------------
    workflow.set_entry_point("retrieve")
    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "generate": "generate",
            "retrieve": "retrieve",
        },
    )
    workflow.add_edge("generate", END)

    return workflow.compile()
