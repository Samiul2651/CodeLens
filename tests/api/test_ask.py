from collections.abc import Iterator

import httpx
import pytest
from fastapi.testclient import TestClient

from codelens.api.app import create_app, get_rag_answerer
from codelens.config import Settings
from codelens.core.models import Answer, AnswerMode, CostRecord


class StubAnswerer:
    def __init__(self, error: Exception | None = None) -> None:
        self.questions: list[tuple[str, str]] = []
        self._error = error

    def answer(self, repo_id: str, question: str) -> Answer:
        self.questions.append((repo_id, question))
        if self._error:
            raise self._error
        return Answer(
            text="Login checks bcrypt [app/auth.py:15-17].",
            mode=AnswerMode.RAG,
            sources=["app/auth.py:15-17", "README.md:1-3"],
            cited=["app/auth.py:15-17"],
            cost=CostRecord("claude-opus-5-5", 1000, 100, 0, 0, 0.006, 1.5),
        )


def client_with(answerer: StubAnswerer) -> Iterator[TestClient]:
    app = create_app(Settings(app_name="Test"))
    app.dependency_overrides[get_rag_answerer] = lambda: answerer
    with TestClient(app) as client:
        yield client


@pytest.fixture
def stub() -> StubAnswerer:
    return StubAnswerer()


@pytest.fixture
def client(stub: StubAnswerer) -> Iterator[TestClient]:
    yield from client_with(stub)


def test_ask_returns_answer_sources_and_cost(client: TestClient, stub: StubAnswerer) -> None:
    response = client.post("/ask", json={"repo_id": "demo", "question": "How does login work?"})

    assert response.status_code == 200
    body = response.json()
    assert body["answer"].startswith("Login checks bcrypt")
    assert body["mode"] == "rag"
    assert body["sources"] == ["app/auth.py:15-17", "README.md:1-3"]
    assert body["cited"] == ["app/auth.py:15-17"]
    assert body["refused"] is False
    assert body["cost"]["cost_usd"] == 0.006
    assert stub.questions == [("demo", "How does login work?")]


@pytest.mark.parametrize(
    "payload",
    [
        {"repo_id": "../etc", "question": "x"},
        {"repo_id": "demo", "question": ""},
        {"repo_id": "demo", "question": "x" * 2001},
        {"repo_id": "demo", "question": "x", "mode": "magic"},
    ],
)
def test_bad_requests_get_422(client: TestClient, payload: dict[str, str]) -> None:
    assert client.post("/ask", json=payload).status_code == 422


def test_long_context_mode_is_not_ready_yet(client: TestClient) -> None:
    response = client.post(
        "/ask", json={"repo_id": "demo", "question": "x", "mode": "long_context"}
    )

    assert response.status_code == 501


def test_backend_down_gives_503_without_internal_details() -> None:
    answerer = StubAnswerer(error=httpx.ConnectError("connection refused to 10.0.0.5"))

    for client in client_with(answerer):
        response = client.post("/ask", json={"repo_id": "demo", "question": "x"})

    assert response.status_code == 503
    assert "10.0.0.5" not in response.text
