"""File hashes per repo, stored as JSON on disk."""

import json
import os
import re
from pathlib import Path

_SAFE_REPO_ID = re.compile(r"^[A-Za-z0-9._-]{1,100}$")


def validate_repo_id(repo_id: str) -> str:
    """Repo ids end up in file names and filters, so allow only a safe set of characters."""
    if not _SAFE_REPO_ID.match(repo_id) or repo_id in {".", ".."}:
        raise ValueError(f"invalid repo id {repo_id!r}: use letters, digits, '.', '_' or '-'")
    return repo_id


class JsonFileStateStore:
    def __init__(self, data_dir: Path) -> None:
        self._dir = data_dir

    def _path(self, repo_id: str) -> Path:
        return self._dir / f"{validate_repo_id(repo_id)}.files.json"

    def load(self, repo_id: str) -> dict[str, str]:
        path = self._path(repo_id)
        if not path.exists():
            return {}
        data: dict[str, str] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def save(self, repo_id: str, hashes: dict[str, str]) -> None:
        """Write to a temp file, then rename: a crash never leaves a half-written file."""
        self._dir.mkdir(parents=True, exist_ok=True)
        path = self._path(repo_id)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(hashes, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(tmp, path)
