import shutil
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from codelens.vectorstore.qdrant import QdrantVectorStore

SAMPLE_REPO = Path(__file__).parent / "fixtures" / "sample_repo"


@pytest.fixture
def repo_copy(tmp_path: Path) -> Path:
    """A writable copy of the sample repo, for tests that change files."""
    target = tmp_path / "repo"
    shutil.copytree(SAMPLE_REPO, target)
    return target


@pytest.fixture
def store() -> QdrantVectorStore:
    """Real Qdrant logic, in memory (no Docker)."""
    return QdrantVectorStore(QdrantClient(location=":memory:"), "test_chunks")
