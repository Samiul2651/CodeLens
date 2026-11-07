import hashlib
from pathlib import Path

from codelens.ingestion.walker import walk_repo
from tests.conftest import SAMPLE_REPO


def test_finds_only_indexable_files() -> None:
    paths = [f.path for f in walk_repo(SAMPLE_REPO, max_file_bytes=500_000)]

    # Skipped: secrets/ and generated.py (.gitignore), node_modules/, blob.json (binary).
    assert paths == ["README.md", "app/auth.py", "payments/PaymentService.cs", "web/cart.js"]


def test_hash_is_sha256_of_file_bytes() -> None:
    auth = next(f for f in walk_repo(SAMPLE_REPO, 500_000) if f.path == "app/auth.py")

    expected = hashlib.sha256((SAMPLE_REPO / "app" / "auth.py").read_bytes()).hexdigest()
    assert auth.sha256 == expected


def test_skips_large_and_non_utf8_files(tmp_path: Path) -> None:
    (tmp_path / "big.py").write_text("x = 1\n" * 100, encoding="utf-8")
    (tmp_path / "latin1.py").write_bytes("name = 'café'\n".encode("latin-1"))
    (tmp_path / "ok.py").write_text("y = 2\n", encoding="utf-8")

    paths = [f.path for f in walk_repo(tmp_path, max_file_bytes=100)]

    assert paths == ["ok.py"]
