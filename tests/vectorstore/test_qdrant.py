import pytest

from codelens.core.models import Chunk
from codelens.vectorstore.qdrant import QdrantVectorStore

A = Chunk("repo1", "a.py", "python", 1, 3, "def a(): ...", "a")
B = Chunk("repo1", "b.py", "python", 5, 9, "def b(): ...")
OTHER_REPO = Chunk("repo2", "a.py", "python", 1, 3, "def a(): ...", "a")


def test_upsert_and_search_round_trip(store: QdrantVectorStore) -> None:
    store.ensure_collection(vector_size=2)
    store.upsert([A, B], [[1.0, 0.0], [0.0, 1.0]])

    results = store.search("repo1", [0.9, 0.1], limit=5)

    assert [r.chunk for r in results] == [A, B]
    assert results[0].score > results[1].score


def test_search_is_limited_to_one_repo(store: QdrantVectorStore) -> None:
    store.ensure_collection(vector_size=2)
    store.upsert([A, OTHER_REPO], [[1.0, 0.0], [1.0, 0.0]])

    assert [r.chunk.repo_id for r in store.search("repo2", [1.0, 0.0], limit=5)] == ["repo2"]


def test_same_chunk_replaces_instead_of_duplicating(store: QdrantVectorStore) -> None:
    store.ensure_collection(vector_size=2)
    store.upsert([A, B], [[1.0, 0.0], [0.0, 1.0]])
    store.upsert([A], [[0.5, 0.5]])

    assert store.count("repo1") == 2


def test_delete_files_only_touches_that_repo(store: QdrantVectorStore) -> None:
    store.ensure_collection(vector_size=2)
    store.upsert([A, B, OTHER_REPO], [[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    store.delete_files("repo1", ["a.py"])

    assert [r.chunk.file_path for r in store.search("repo1", [1.0, 0.0], 5)] == ["b.py"]
    assert store.count("repo2") == 1


def test_wrong_vector_size_is_refused(store: QdrantVectorStore) -> None:
    store.ensure_collection(vector_size=2)
    store.ensure_collection(vector_size=2)  # same size: fine

    with pytest.raises(ValueError, match="vector size"):
        store.ensure_collection(vector_size=3)


def test_delete_count_and_search_before_collection_exists(store: QdrantVectorStore) -> None:
    store.delete_files("repo1", ["a.py"])

    assert store.count("repo1") == 0
    assert store.search("repo1", [1.0, 0.0], limit=5) == []


def test_all_chunks_reads_every_page_in_file_order() -> None:
    from qdrant_client import QdrantClient

    store = QdrantVectorStore(QdrantClient(location=":memory:"), "paged", page_size=2)
    store.ensure_collection(vector_size=2)
    chunks = [Chunk("repo1", f"f{n % 3}.py", "python", n, n, f"line {n}") for n in range(1, 8)]
    store.upsert(chunks, [[1.0, 0.0]] * len(chunks))
    store.upsert([OTHER_REPO], [[1.0, 0.0]])

    result = store.all_chunks("repo1")

    assert len(result) == 7
    assert result == sorted(chunks, key=lambda c: (c.file_path, c.start_line))
    assert store.all_chunks("missing-repo") == []


def test_repo_ids_lists_each_repo_once_sorted(store: QdrantVectorStore) -> None:
    assert store.repo_ids() == []  # no collection yet

    store.ensure_collection(vector_size=2)
    store.upsert([OTHER_REPO, A, B], [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])

    assert store.repo_ids() == ["repo1", "repo2"]
