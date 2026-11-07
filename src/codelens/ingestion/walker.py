"""Find the files in a repo that are worth indexing."""

import hashlib
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pathspec

from codelens.chunking.languages import detect_language

# Never project code, even when a repo forgets to ignore them.
IGNORED_DIRS: frozenset[str] = frozenset(
    {".git", "node_modules", "venv", ".venv", "__pycache__", "bin", "obj", "dist", "build",
     ".mypy_cache", ".pytest_cache", ".ruff_cache", ".idea", ".vscode", ".codelens"}
)  # fmt: skip
BINARY_SNIFF_BYTES = 8192


@dataclass(frozen=True)
class RepoFile:
    path: str  # relative to the repo root, with "/" separators
    text: str
    sha256: str


def _load_gitignore(root: Path) -> pathspec.GitIgnoreSpec:
    gitignore = root / ".gitignore"
    patterns = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.is_file() else []
    return pathspec.GitIgnoreSpec.from_lines(patterns)


def _looks_binary(data: bytes) -> bool:
    return b"\x00" in data[:BINARY_SNIFF_BYTES]


def walk_repo(root: Path, max_file_bytes: int) -> Iterator[RepoFile]:
    """Yield indexable text files, sorted by path.

    Skipped: ignored folders, paths matched by the root .gitignore, unknown file types,
    files over max_file_bytes, binary files, and files that are not valid UTF-8.
    """
    root = root.resolve()
    gitignore = _load_gitignore(root)
    # Sort by the posix path text: Path objects sort case-insensitively on Windows only.
    for path in sorted(root.rglob("*"), key=lambda p: p.relative_to(root).as_posix()):
        if not path.is_file() or path.is_symlink():
            continue
        relative = path.relative_to(root)
        rel_posix = relative.as_posix()
        if any(part in IGNORED_DIRS for part in relative.parts[:-1]):
            continue
        if gitignore.match_file(rel_posix) or detect_language(rel_posix) is None:
            continue
        if path.stat().st_size > max_file_bytes:
            continue
        data = path.read_bytes()
        if _looks_binary(data):
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        yield RepoFile(path=rel_posix, text=text, sha256=hashlib.sha256(data).hexdigest())
