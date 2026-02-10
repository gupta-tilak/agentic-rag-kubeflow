"""Agent state definition — shared across all graph nodes."""

from __future__ import annotations

from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Typed state that flows through the LangGraph agent.

    Attributes
    ----------
    messages:
        Conversation history managed by ``add_messages`` reducer.
    documents:
        Retrieved context documents relevant to the current query.
    query:
        The user's current natural-language question.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    documents: list[Document]
    query: str
