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


def test_delete_and_count_before_collection_exists(store: QdrantVectorStore) -> None:
    store.delete_files("repo1", ["a.py"])

    assert store.count("repo1") == 0
