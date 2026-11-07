"""Index a repo: walk -> chunk changed files -> embed in batches -> store.

Only files whose content hash changed since the last run are re-chunked and re-embedded.
Deleted files lose their chunks.

Crash safety: file hashes are saved only after every batch is stored. If a run fails
halfway, the next run sees the same files as changed and does them again; stable point
ids make that safe (same chunk -> same id -> overwritten, not duplicated).
"""

from dataclasses import dataclass
from pathlib import Path

from codelens.core.interfaces import Chunker, Embedder, FileStateStore, VectorStore
from codelens.core.models import Chunk
from codelens.ingestion.state import validate_repo_id
from codelens.ingestion.walker import walk_repo


@dataclass(frozen=True)
class IngestReport:
    files_seen: int
    files_changed: int  # new or modified
    files_deleted: int
    chunks_written: int
    embed_requests: int


def embedding_text(chunk: Chunk) -> str:
    """The code plus where it lives, so 'where is login handled?' can match
    the file path and symbol name, not only the code."""
    header = f"file: {chunk.file_path}"
    if chunk.symbol:
        header += f"\nsymbol: {chunk.symbol}"
    return f"{header}\n\n{chunk.text}"


class IngestionService:
    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        store: VectorStore,
        state: FileStateStore,
        batch_size: int,
        max_file_bytes: int,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        self._chunker = chunker
        self._embedder = embedder
        self._store = store
        self._state = state
        self._batch_size = batch_size
        self._max_file_bytes = max_file_bytes

    def ingest(self, repo_id: str, root: Path) -> IngestReport:
        validate_repo_id(repo_id)
        if not root.is_dir():
            raise NotADirectoryError(root)

        previous = self._state.load(repo_id)
        files = list(walk_repo(root, self._max_file_bytes))
        current = {f.path: f.sha256 for f in files}
        changed = [f for f in files if previous.get(f.path) != f.sha256]
        deleted = sorted(set(previous) - set(current))

        # Old chunks of changed and deleted files go first: line numbers may have moved,
        # so their old point ids would not be overwritten.
        stale = [f.path for f in changed if f.path in previous] + deleted
        self._store.delete_files(repo_id, stale)

        chunks = [c for f in changed for c in self._chunker.chunk_file(repo_id, f.path, f.text)]
        embed_requests = self._embed_and_store(chunks)

        self._state.save(repo_id, current)
        return IngestReport(
            files_seen=len(files),
            files_changed=len(changed),
            files_deleted=len(deleted),
            chunks_written=len(chunks),
            embed_requests=embed_requests,
        )

    def _embed_and_store(self, chunks: list[Chunk]) -> int:
        requests = 0
        for start in range(0, len(chunks), self._batch_size):
            batch = chunks[start : start + self._batch_size]
            vectors = self._embedder.embed([embedding_text(c) for c in batch])  # one request
            requests += 1
            if start == 0:
                self._store.ensure_collection(vector_size=len(vectors[0]))
            self._store.upsert(batch, vectors)  # one write
        return requests
