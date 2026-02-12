"""LangGraph graph definition — the agentic RAG workflow.

This module wires the nodes defined in :mod:`agentic_rag.agent.nodes`
into a compiled :class:`StateGraph` that implements **true** agentic
retrieval-augmented generation:

1. **Analyse** the user's question (intent, key concepts, tool plan).
2. **Execute** the planned retrieval tools (vector search, keyword
   search, document lookup, web search, …).
3. **Grade** whether the evidence is sufficient.
4. **Loop** — if more information is needed, refine the query and
   retrieve again (up to ``max_iterations``).
5. **Synthesise** a final answer with inline citations.

The graph can be tested locally without Kubeflow or any external
infrastructure by injecting fake tools / LLM stubs (see tests).
"""

from __future__ import annotations

from typing import Any

from langgraph.graph import END, StateGraph

from agentic_rag.agent.nodes import (
    analyze_query,
    execute_tools,
    grade_results,
    should_continue,
    synthesize,
)
from agentic_rag.agent.state import AgentState


def build_graph() -> StateGraph:
    """Construct and return the compiled LangGraph agent.

    Graph topology::

        ┌─────────┐
        │  START   │
        └────┬─────┘
             ▼
      ┌──────────────┐
      │ analyze_query │   ← intent / key concepts / tool plan
      └──────┬───────┘
             ▼
      ┌──────────────┐
      │ execute_tools │◄──────────────────┐
      └──────┬───────┘                    │
             ▼                            │
      ┌──────────────┐   needs_more_info  │
      │ grade_results ├───────────────────┘
      └──────┬───────┘
             │ sufficient
             ▼
      ┌──────────────┐
      │  synthesize   │
      └──────┬───────┘
             ▼
          [ END ]

    Returns
    -------
    CompiledGraph
        A compiled LangGraph workflow ready for ``.invoke()``.
    """
    workflow = StateGraph(AgentState)

    # -- Nodes ---------------------------------------------------------------
    workflow.add_node("analyze_query", analyze_query)
    workflow.add_node("execute_tools", execute_tools)
    workflow.add_node("grade_results", grade_results)
    workflow.add_node("synthesize", synthesize)

    # -- Edges ---------------------------------------------------------------
    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "execute_tools")
    workflow.add_edge("execute_tools", "grade_results")

    # Conditional: loop back to execute_tools or proceed to synthesize
    workflow.add_conditional_edges(
        "grade_results",
        should_continue,
        {
            "execute_tools": "execute_tools",
            "synthesize": "synthesize",
        },
    )
    workflow.add_edge("synthesize", END)

    return workflow.compile()


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------


def create_initial_state(query: str, *, max_iterations: int = 3) -> dict[str, Any]:
    """Build a minimal initial state dict for ``graph.invoke()``.

    Usage::

        graph = build_graph()
        state = create_initial_state("How does KServe autoscaling work?")
        result = graph.invoke(state)
        print(result["answer"])
    """
    return {
        "query": query,
        "messages": [],
        "query_analysis": {},
        "retrieval_plan": [],
        "documents": [],
        "reasoning_trace": [],
        "tool_calls_made": [],
        "citations": [],
        "answer": "",
        "iteration": 0,
        "max_iterations": max_iterations,
        "needs_more_info": False,
    }
