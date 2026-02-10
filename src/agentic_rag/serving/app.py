"""FastAPI application exposing the RAG agent as a REST API."""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

from agentic_rag.agent.graph import build_graph

app = FastAPI(
    title="Agentic RAG API",
    version="0.1.0",
    description="REST interface to the Agentic RAG LangGraph agent.",
)


# ── Request / Response schemas ────────────────────────────────────────
class QueryRequest(BaseModel):
    """Incoming question from the user."""

    query: str


class QueryResponse(BaseModel):
    """Answer returned by the agent."""

    answer: str
    sources: list[str] = []


# ── Routes ────────────────────────────────────────────────────────────
@app.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe."""
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Run the RAG agent and return an answer."""
    graph = build_graph()
    result = graph.invoke({"query": request.query, "messages": [], "documents": []})

    answer = result["messages"][-1].content if result.get("messages") else ""
    sources = [doc.metadata.get("source", "") for doc in result.get("documents", [])]

    return QueryResponse(answer=answer, sources=sources)
