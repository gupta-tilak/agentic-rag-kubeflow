"""Agent state definition — shared across all graph nodes.

The state is the *single source of truth* that flows through every node
in the LangGraph agent.  Each field is documented so that new nodes can
be added without guessing what data is available.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Annotated, Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


# ---------------------------------------------------------------------------
# Structured sub-models (plain dataclasses — no Pydantic required in state)
# ---------------------------------------------------------------------------


@dataclass
class ReasoningStep:
    """One atomic reasoning step the agent performed.

    Attributes
    ----------
    node:
        The graph node that produced this step (e.g. ``"analyze_query"``).
    thought:
        The agent's internal reasoning at this point.
    action:
        What the agent decided to do next.
    observation:
        The outcome / evidence gathered by that action (may be empty until
        the action is actually executed).
    """

    node: str
    thought: str
    action: str = ""
    observation: str = ""


@dataclass
class ToolCall:
    """Record of a single tool invocation.

    Attributes
    ----------
    tool_name:
        Which tool was called (e.g. ``"vector_search"``).
    tool_input:
        The arguments passed to the tool.
    result_summary:
        Short summary of the tool output.
    documents_returned:
        Number of documents the tool produced.
    """

    tool_name: str
    tool_input: dict[str, Any] = field(default_factory=dict)
    result_summary: str = ""
    documents_returned: int = 0


@dataclass
class SourceCitation:
    """A structured citation attached to a claim in the final answer.

    Attributes
    ----------
    citation_id:
        Short reference id, e.g. ``"[1]"``.
    source:
        Human-readable source locator (file path, URL, …).
    chunk_index:
        Ordinal position of the chunk within the source document.
    score:
        Relevance score from the retriever.
    excerpt:
        Short excerpt from the source that backs the claim.
    """

    citation_id: str
    source: str
    chunk_index: int | None = None
    score: float | None = None
    excerpt: str = ""


# ---------------------------------------------------------------------------
# Reducers
# ---------------------------------------------------------------------------


def _append_list(existing: list[Any], new: list[Any]) -> list[Any]:
    """Reducer that appends *new* items to the *existing* list."""
    return existing + new


# ---------------------------------------------------------------------------
# Agent state
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """Typed state that flows through the LangGraph agent.

    Attributes
    ----------
    messages:
        Conversation history managed by LangGraph's ``add_messages`` reducer.
    query:
        The user's current natural-language question.
    query_analysis:
        Structured analysis of the query produced by the ``analyze_query``
        node (intent, key concepts, suggested tools, …).
    retrieval_plan:
        A list of ``{"tool": ..., "query": ..., "reason": ...}`` dicts
        describing which retrieval tools to call and why.
    documents:
        All context documents gathered across tool calls, deduplicated.
    reasoning_trace:
        Ordered list of :class:`ReasoningStep` objects that make the
        agent's multi-step reasoning **visible** and auditable.
    tool_calls_made:
        Chronological log of every tool invocation.
    citations:
        Structured citations extracted during synthesis.
    answer:
        The final synthesised answer (populated by the ``synthesize`` node).
    iteration:
        Current loop iteration (starts at 0).
    max_iterations:
        Safety cap to prevent runaway loops (default: 3).
    needs_more_info:
        Flag set by the ``grade_results`` node when gathered evidence is
        insufficient and another retrieval round is warranted.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    query_analysis: dict[str, Any]
    retrieval_plan: list[dict[str, Any]]
    documents: list[Document]
    reasoning_trace: Annotated[list[ReasoningStep], _append_list]
    tool_calls_made: Annotated[list[ToolCall], _append_list]
    citations: list[SourceCitation]
    answer: str
    iteration: int
    max_iterations: int
    needs_more_info: bool
