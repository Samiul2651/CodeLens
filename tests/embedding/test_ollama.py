import json

import httpx
import pytest

from codelens.embedding.ollama import OllamaEmbedder


def make_embedder(requests: list[httpx.Request], extra_vectors: int = 0) -> OllamaEmbedder:
    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        count = len(json.loads(request.content)["input"]) + extra_vectors
        return httpx.Response(200, json={"embeddings": [[0.5, 0.5]] * count})

    http = httpx.Client(transport=httpx.MockTransport(handler))
    return OllamaEmbedder("http://ollama:11434/", "nomic-embed-text", http=http)


def test_one_request_for_the_whole_list() -> None:
    requests: list[httpx.Request] = []

    vectors = make_embedder(requests).embed(["a", "b", "c"])

    assert len(requests) == 1
    assert str(requests[0].url) == "http://ollama:11434/api/embed"
    assert json.loads(requests[0].content) == {
        "model": "nomic-embed-text",
        "input": ["a", "b", "c"],
    }
    assert vectors == [[0.5, 0.5]] * 3


def test_empty_input_sends_nothing() -> None:
    requests: list[httpx.Request] = []

    assert make_embedder(requests).embed([]) == []
    assert requests == []


def test_vector_count_mismatch_is_an_error() -> None:
    with pytest.raises(RuntimeError):
        make_embedder([], extra_vectors=1).embed(["a"])
