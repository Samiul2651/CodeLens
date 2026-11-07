from pathlib import Path

import pytest

from codelens.ingestion.state import JsonFileStateStore, validate_repo_id


def test_round_trip(tmp_path: Path) -> None:
    store = JsonFileStateStore(tmp_path / "data")

    assert store.load("repo") == {}
    store.save("repo", {"a.py": "abc", "b.py": "def"})

    assert store.load("repo") == {"a.py": "abc", "b.py": "def"}
    assert not list((tmp_path / "data").glob("*.tmp")), "temp file must be renamed away"


@pytest.mark.parametrize("repo_id", ["", "..", "../etc", "a/b", "a b", "x" * 101])
def test_unsafe_repo_ids_are_rejected(repo_id: str) -> None:
    with pytest.raises(ValueError):
        validate_repo_id(repo_id)


def test_normal_repo_ids_are_fine() -> None:
    assert validate_repo_id("fastapi-basics_v2.0") == "fastapi-basics_v2.0"
