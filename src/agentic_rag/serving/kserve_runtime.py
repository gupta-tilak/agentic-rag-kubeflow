"""KServe custom model runtime for the RAG agent."""

from __future__ import annotations

from typing import Any

import kserve

from agentic_rag.agent.graph import build_graph


class RAGAgentModel(kserve.Model):
    """KServe-compatible model that wraps the LangGraph agent.

    This class implements the ``predict`` interface expected by KServe
    so the agent can be deployed as an ``InferenceService``.
    """

    def __init__(self, name: str = "rag-agent") -> None:
        super().__init__(name)
        self.graph = None
        self.ready = False

    def load(self) -> None:
        """Compile the LangGraph agent (called once at startup)."""
        self.graph = build_graph()
        self.ready = True

    def predict(self, payload: dict[str, Any], headers: dict[str, str] | None = None) -> dict:
        """Run inference — called on every request.

        Parameters
        ----------
        payload:
            ``{"instances": [{"query": "..."}]}`` following the v2 protocol.
        headers:
            Optional HTTP headers.

        Returns
        -------
        dict
            ``{"predictions": [{"answer": "...", "sources": [...]}]}``
        """
        instances = payload.get("instances", [])
        predictions = []

        for instance in instances:
            query = instance.get("query", "")
            result = self.graph.invoke({"query": query, "messages": [], "documents": []})

            answer = result["messages"][-1].content if result.get("messages") else ""
            sources = [doc.metadata.get("source", "") for doc in result.get("documents", [])]
            predictions.append({"answer": answer, "sources": sources})

        return {"predictions": predictions}


if __name__ == "__main__":
    model = RAGAgentModel()
    model.load()
    kserve.ModelServer().start([model])
