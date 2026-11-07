from pathlib import Path

import pytest

from codelens.answering.context_packer import ContextPacker
from codelens.answering.long_context import NOTHING_FITS, LongContextAnswerer
from codelens.answering.prompts import SYSTEM_PROMPT
from codelens.answering.rag import NOTHING_INDEXED
from codelens.chunking.chunkers import TreeSitterChunker
from codelens.core.models import AnswerMode
from codelens.ingestion.service import IngestionService
from codelens.vectorstore.qdrant import QdrantVectorStore
from tests.fakes import (
    BagOfWordsEmbedder,
    CharTokenCounter,
    FakeLanguageModel,
    InMemoryStateStore,
    ListCostLog,
)


@pytest.fixture
def indexed(repo_copy: Path, store: QdrantVectorStore) -> QdrantVectorStore:
    IngestionService(
        TreeSitterChunker(40), BagOfWordsEmbedder(), store, InMemoryStateStore(), 16, 500_000
    ).ingest("demo", repo_copy)
    return store


def make_answerer(
    store: QdrantVectorStore, model: FakeLanguageModel, log: ListCostLog, budget: int = 100_000
) -> LongContextAnswerer:
    return LongContextAnswerer(store, model, ContextPacker(CharTokenCounter(), budget), log)


def test_whole_repo_goes_in_the_cached_block(indexed: QdrantVectorStore) -> None:
    model, log = FakeLanguageModel("Login uses bcrypt [app/auth.py:15-17]."), ListCostLog()

    answer = make_answerer(indexed, model, log).answer("demo", "How does login work?")

    [(system, prompt)] = model.calls
    [context] = model.cached_contexts
    assert system == SYSTEM_PROMPT
    assert prompt == "Question: How does login work?", "question stays out of the cached block"
    assert context is not None and context.startswith("<excerpts>")
    assert context.index('source="README.md') < context.index('source="app/auth.py')
    assert answer.mode is AnswerMode.LONG_CONTEXT
    assert len(answer.sources) == indexed.count("demo"), "every chunk was shown"
    assert answer.cited == ["app/auth.py:15-17"]
    assert not answer.context_truncated
    [entry] = log.entries
    assert (entry.label, entry.repo_id) == ("long_context", "demo")


def test_same_repo_gives_identical_cached_block(indexed: QdrantVectorStore) -> None:
    model = FakeLanguageModel()
    answerer = make_answerer(indexed, model, ListCostLog())

    answerer.answer("demo", "First question?")
    answerer.answer("demo", "A different question?")

    first, second = model.cached_contexts
    assert first == second, "any change in the cached block would be a cache miss"


def test_small_budget_is_reported_as_truncated(indexed: QdrantVectorStore) -> None:
    answer = make_answerer(indexed, FakeLanguageModel(), ListCostLog(), budget=700).answer(
        "demo", "Q?"
    )

    assert answer.context_truncated
    assert 0 < len(answer.sources) < indexed.count("demo")


def test_nothing_indexed_or_nothing_fits(indexed: QdrantVectorStore) -> None:
    model, log = FakeLanguageModel(), ListCostLog()

    unknown = make_answerer(indexed, model, log).answer("other-repo", "Q?")
    too_small = make_answerer(indexed, model, log, budget=5).answer("demo", "Q?")

    assert unknown.text == NOTHING_INDEXED
    assert too_small.text == NOTHING_FITS and too_small.context_truncated
    assert model.calls == [] and log.entries == []


def test_refusal(indexed: QdrantVectorStore) -> None:
    log = ListCostLog()

    answer = make_answerer(indexed, FakeLanguageModel(refused=True), log).answer("demo", "Q?")

    assert answer.refused and "declined" in answer.text
    assert len(log.entries) == 1
