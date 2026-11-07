import math
from pathlib import Path

import pytest

from codelens.chunking.chunkers import TreeSitterChunker
from codelens.core.models import Chunk
from codelens.ingestion.service import IngestionService, embedding_text
from codelens.vectorstore.qdrant import QdrantVectorStore
from tests.fakes import BagOfWordsEmbedder, InMemoryStateStore


def make_service(
    store: QdrantVectorStore, embedder: BagOfWordsEmbedder, state: InMemoryStateStore,
    batch_size: int = 4,
) -> IngestionService:  # fmt: skip
    return IngestionService(
        chunker=TreeSitterChunker(max_lines=40),
        embedder=embedder,
        store=store,
        state=state,
        batch_size=batch_size,
        max_file_bytes=500_000,
    )


def test_first_run_indexes_everything_in_batches(repo_copy: Path, store: QdrantVectorStore) -> None:
    embedder = BagOfWordsEmbedder()

    report = make_service(store, embedder, InMemoryStateStore()).ingest("demo", repo_copy)

    assert report.files_seen == 4
    assert report.files_changed == 4
    assert report.embed_requests == math.ceil(report.chunks_written / 4) == len(embedder.requests)
    assert all(len(batch) <= 4 for batch in embedder.requests)
    assert store.count("demo") == report.chunks_written


def test_second_run_without_changes_does_nothing(repo_copy: Path, store: QdrantVectorStore) -> None:
    embedder, state = BagOfWordsEmbedder(), InMemoryStateStore()
    service = make_service(store, embedder, state)
    first = service.ingest("demo", repo_copy)

    second = service.ingest("demo", repo_copy)

    assert (second.files_changed, second.chunks_written, second.embed_requests) == (0, 0, 0)
    assert store.count("demo") == first.chunks_written


def test_changed_file_is_redone_and_its_old_chunks_removed(
    repo_copy: Path, store: QdrantVectorStore
) -> None:
    embedder, state = BagOfWordsEmbedder(), InMemoryStateStore()
    service = make_service(store, embedder, state)
    service.ingest("demo", repo_copy)
    cart = repo_copy / "web" / "cart.js"
    # Two new lines at the top move every function down, so old point ids no longer match.
    cart.write_text("// header\n// header\n" + cart.read_text(encoding="utf-8"), encoding="utf-8")

    report = service.ingest("demo", repo_copy)

    assert report.files_changed == 1
    assert {t.split("\n")[0] for t in embedder.requests[-1]} == {"file: web/cart.js"}
    cart_chunks = [
        r.chunk
        for r in store.search("demo", embedder.vector("cart"), limit=100)
        if r.chunk.file_path == "web/cart.js"
    ]
    assert len(cart_chunks) == report.chunks_written, "old cart.js chunks must be gone"


def test_deleted_file_loses_its_chunks(repo_copy: Path, store: QdrantVectorStore) -> None:
    embedder, state = BagOfWordsEmbedder(), InMemoryStateStore()
    service = make_service(store, embedder, state)
    service.ingest("demo", repo_copy)
    (repo_copy / "app" / "auth.py").unlink()

    report = service.ingest("demo", repo_copy)

    assert report.files_deleted == 1
    paths = {r.chunk.file_path for r in store.search("demo", embedder.vector("x"), limit=100)}
    assert "app/auth.py" not in paths
    assert "app/auth.py" not in state.load("demo")


def test_hashes_are_not_saved_when_storing_fails(repo_copy: Path, store: QdrantVectorStore) -> None:
    class BrokenEmbedder(BagOfWordsEmbedder):
        def embed(self, texts: list[str]) -> list[list[float]]:
            raise ConnectionError("Ollama is down")

    state = InMemoryStateStore()

    with pytest.raises(ConnectionError):
        make_service(store, BrokenEmbedder(), state).ingest("demo", repo_copy)

    assert state.load("demo") == {}, "next run must redo these files"


def test_rejects_bad_repo_id_and_missing_folder(tmp_path: Path, store: QdrantVectorStore) -> None:
    service = make_service(store, BagOfWordsEmbedder(), InMemoryStateStore())

    with pytest.raises(ValueError):
        service.ingest("../evil", tmp_path)
    with pytest.raises(NotADirectoryError):
        service.ingest("demo", tmp_path / "missing")


def test_embedding_text_has_path_and_symbol() -> None:
    chunk = Chunk("r", "app/auth.py", "python", 1, 2, "code", "AuthService.login")

    assert embedding_text(chunk) == "file: app/auth.py\nsymbol: AuthService.login\n\ncode"
