"""Unit tests for the serving layer."""

from fastapi.testclient import TestClient


def test_health_endpoint() -> None:
    """GET /health should return 200 with status ok."""
    from agentic_rag.serving.app import app

    client = TestClient(app)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
