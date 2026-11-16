"""API tests. Require the backend dependencies:

    pip install -r requirements-dev.txt
    pytest -v
"""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)
ALLOWED = "https://srinithya2205-tech.github.io"


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["portfolio_loaded"] is True


def test_portfolio_returns_source_of_truth():
    r = client.get("/portfolio")
    assert r.status_code == 200
    body = r.json()
    assert body["profile"]["name"] == "Sri Nithya"
    assert len(body["projects"]) == 3
    assert len(body["certifications"]) == 2


def test_ask_answers_from_portfolio():
    r = client.post("/ask", json={"question": "What projects has Sri Nithya built?"})
    assert r.status_code == 200
    body = r.json()
    assert body["matched"] is True
    assert "Air Quality Index (AQI) Analysis" in body["answer"]
    assert set(body) == {"answer", "matched", "topics", "sources", "suggestions"}


def test_ask_does_not_invent_experience():
    r = client.post("/ask", json={"question": "Where has she worked?"})
    assert r.status_code == 200
    assert "doesn't list any work or internship experience" in r.json()["answer"]


def test_ask_out_of_scope_is_flagged():
    r = client.post("/ask", json={"question": "What's the weather in Paris?"})
    assert r.status_code == 200
    assert r.json()["matched"] is False


def test_ask_rejects_empty_question():
    assert client.post("/ask", json={"question": ""}).status_code == 422
    assert client.post("/ask", json={"question": "     "}).status_code == 422


def test_ask_rejects_missing_field_and_wrong_type():
    assert client.post("/ask", json={}).status_code == 422
    assert client.post("/ask", json={"question": 123}).status_code == 422


def test_ask_rejects_too_long_question():
    assert client.post("/ask", json={"question": "a" * 501}).status_code == 422


def test_cors_preflight_allows_portfolio_origin():
    r = client.options(
        "/ask",
        headers={
            "Origin": ALLOWED,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert r.status_code == 200
    assert r.headers.get("access-control-allow-origin") == ALLOWED


def test_cors_blocks_unknown_origin():
    r = client.get("/health", headers={"Origin": "https://evil.example"})
    assert "access-control-allow-origin" not in r.headers
