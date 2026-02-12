"""
Agent — autonomous LLM agent built with LangGraph.

This module contains **zero** infrastructure dependencies.  It wires
together tools (retrieval, web search, …) into a LangGraph state-machine
that can be tested locally without Kubeflow or KServe.

Public API
----------
- :func:`build_graph` — compile the agentic workflow.
- :func:`create_initial_state` — bootstrap the state dict for ``graph.invoke()``.
- :class:`AgentState` — the TypedDict flowing through every node.
"""

from agentic_rag.agent.graph import build_graph, create_initial_state
from agentic_rag.agent.state import AgentState, ReasoningStep, SourceCitation, ToolCall

__all__ = [
    "AgentState",
    "ReasoningStep",
    "SourceCitation",
    "ToolCall",
    "build_graph",
    "create_initial_state",
]
