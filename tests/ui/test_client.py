import json

import httpx
import pytest

from codelens.core.models import AnswerMode
from codelens.ui.client import ApiError, CodeLensClient

ANSWER = {
    "answer": "Ids come from a counter [app/store.py:10-14].",
    "mode": "rag",
    "sources": ["app/store.py:10-14"],
    "cited": ["app/store.py:10-14"],
    "refused": False,
    "context_truncated": False,
    "cost": None,
}


def client_answering(handler: httpx.MockTransport) -> CodeLensClient:
    return CodeLensClient(httpx.Client(base_url="http://api.test", transport=handler))


def test_repos_reads_the_list() -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(200, json={"repos": ["a", "b"]}))

    assert client_answering(transport).repos() == ["a", "b"]


def test_ask_sends_one_request_and_parses_the_answer() -> None:
    sent: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, json=ANSWER)

    response = client_answering(httpx.MockTransport(handler)).ask(
        "demo", "how are ids made?", AnswerMode.LONG_CONTEXT
    )

    assert response.cited == ["app/store.py:10-14"]
    assert len(sent) == 1
    assert sent[0].url.path == "/ask"
    assert json.loads(sent[0].content) == {
        "repo_id": "demo",
        "question": "how are ids made?",
        "mode": "long_context",
    }


def test_bad_question_is_rejected_without_a_call() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("no request expected")

    with pytest.raises(ApiError, match="did not accept"):
        client_answering(httpx.MockTransport(handler)).ask("../etc", "x", AnswerMode.RAG)


@pytest.mark.parametrize(
    ("status", "message"),
    [(503, "not reachable"), (502, "API log"), (500, "HTTP 500")],
)
def test_error_status_becomes_a_plain_message(status: int, message: str) -> None:
    transport = httpx.MockTransport(lambda r: httpx.Response(status, text="secret 10.0.0.5"))

    with pytest.raises(ApiError, match=message) as caught:
        client_answering(transport).repos()
    assert "10.0.0.5" not in str(caught.value)


def test_plain_api_reason_is_shown() -> None:
    reason = {"detail": "Claude credentials are not set: add ANTHROPIC_API_KEY to .env"}
    transport = httpx.MockTransport(lambda r: httpx.Response(503, json=reason))

    with pytest.raises(ApiError, match="ANTHROPIC_API_KEY"):
        client_answering(transport).repos()


def test_api_down_says_where_it_looked() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(ApiError, match="http://api.test"):
        client_answering(httpx.MockTransport(handler)).repos()
