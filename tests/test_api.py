import pytest
from fastapi.testclient import TestClient

from recon.agent import Agent
from recon.api import app, get_agent
from recon.llm import MockModel
from recon.tracing import TraceStore


@pytest.fixture
def client(tmp_path):
    mock_agent = Agent(model=MockModel(), store=TraceStore(tmp_path))
    app.dependency_overrides[get_agent] = lambda: mock_agent
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_ask_then_fetch_trace(client):
    r = client.post("/ask", json={"question": "Is invoice DUP-2026-0001 from Dupont fully paid?"})
    assert r.status_code == 200
    body = r.json()
    assert body["outcome"] == "answered" and body["answer"]["invoice_number"] == "DUP-2026-0001"
    t = client.get(body["trace_url"])
    assert t.status_code == 200 and t.json()["model"] == "mock"


def test_rejects_empty_question(client):
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_unknown_trace_is_404(client):
    assert client.get("/trace/nope").status_code == 404
