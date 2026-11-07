from pathlib import Path

import pytest

from codelens.answering.prompts import SYSTEM_PROMPT
from codelens.answering.rag import NOTHING_INDEXED, RagAnswerer
from codelens.chunking.chunkers import TreeSitterChunker
from codelens.core.models import AnswerMode
from codelens.ingestion.service import IngestionService
from codelens.vectorstore.qdrant import QdrantVectorStore
from tests.fakes import BagOfWordsEmbedder, FakeLanguageModel, InMemoryStateStore, ListCostLog


@pytest.fixture
def indexed(repo_copy: Path, store: QdrantVectorStore) -> BagOfWordsEmbedder:
    """The sample repo, ingested into the in-memory store as repo "demo"."""
    embedder = BagOfWordsEmbedder()
    IngestionService(
        TreeSitterChunker(40), embedder, store, InMemoryStateStore(), 16, 500_000
    ).ingest("demo", repo_copy)
    return embedder


def make_answerer(
    embedder: BagOfWordsEmbedder, store: QdrantVectorStore, model: FakeLanguageModel,
    log: ListCostLog, top_k: int = 3,
) -> RagAnswerer:  # fmt: skip
    return RagAnswerer(embedder, store, model, log, top_k)


def test_answers_from_the_closest_chunks(
    indexed: BagOfWordsEmbedder, store: QdrantVectorStore
) -> None:
    model = FakeLanguageModel("Refunds call Stripe [payments/PaymentService.cs:13-16].")
    log = ListCostLog()
    requests_before = len(indexed.requests)

    answer = make_answerer(indexed, store, model, log).answer(
        "demo", "how is a refund paymentId handled"
    )

    assert len(indexed.requests) == requests_before + 1, "question embedded in one request"
    [(system, prompt)] = model.calls
    assert system == SYSTEM_PROMPT
    assert "PaymentService.Refund" in prompt
    assert prompt.endswith("Question: how is a refund paymentId handled")
    assert answer.mode is AnswerMode.RAG
    assert len(answer.sources) == 3
    assert answer.sources[0] == "payments/PaymentService.cs:13-16"
    assert answer.cited == ["payments/PaymentService.cs:13-16"]
    assert answer.cost is not None and answer.cost.cost_usd == pytest.approx(0.006)


def test_every_call_is_logged(indexed: BagOfWordsEmbedder, store: QdrantVectorStore) -> None:
    log = ListCostLog()

    make_answerer(indexed, store, FakeLanguageModel(), log).answer("demo", "password")

    [entry] = log.entries
    assert (entry.label, entry.repo_id, entry.question) == ("rag", "demo", "password")


def test_unknown_repo_means_no_model_call(
    indexed: BagOfWordsEmbedder, store: QdrantVectorStore
) -> None:
    model, log = FakeLanguageModel(), ListCostLog()

    answer = make_answerer(indexed, store, model, log).answer("other-repo", "password")

    assert answer.text == NOTHING_INDEXED
    assert model.calls == [] and log.entries == []
    assert answer.cost is None


def test_refusal(indexed: BagOfWordsEmbedder, store: QdrantVectorStore) -> None:
    log = ListCostLog()

    answer = make_answerer(indexed, store, FakeLanguageModel(refused=True), log).answer(
        "demo", "password"
    )

    assert answer.refused
    assert "declined" in answer.text and "cyber" in answer.text
    assert len(log.entries) == 1, "refused calls cost money too"


def test_top_k_must_be_positive(store: QdrantVectorStore) -> None:
    with pytest.raises(ValueError):
        RagAnswerer(BagOfWordsEmbedder(), store, FakeLanguageModel(), ListCostLog(), top_k=0)
