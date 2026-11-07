"""The parts of CodeLens, as Protocols. Real versions live in their own packages;
tests use fakes. Everything else depends on these, never on a concrete class.
"""

from typing import Protocol

from codelens.core.models import Answer, Chunk, Completion, CostLogEntry, RetrievedChunk


class Chunker(Protocol):
    def chunk_file(self, repo_id: str, file_path: str, text: str) -> list[Chunk]: ...


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        """One request for the whole list. Callers batch large inputs."""
        ...


class VectorStore(Protocol):
    def ensure_collection(self, vector_size: int) -> None: ...

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """One write for the whole batch."""
        ...

    def delete_files(self, repo_id: str, file_paths: list[str]) -> None:
        """Remove every chunk of these files, in one call."""
        ...

    def search(self, repo_id: str, vector: list[float], limit: int) -> list[RetrievedChunk]: ...

    def all_chunks(self, repo_id: str) -> list[Chunk]:
        """Every chunk of a repo, sorted by file path, then start line."""
        ...


class FileStateStore(Protocol):
    """Remembers a content hash per file, so re-indexing only touches changed files."""

    def load(self, repo_id: str) -> dict[str, str]: ...

    def save(self, repo_id: str, hashes: dict[str, str]) -> None: ...


class LanguageModel(Protocol):
    def complete(self, system: str, prompt: str, cached_context: str | None = None) -> Completion:
        """One call. Returns the answer text plus tokens, cost and latency.

        cached_context: a large, repeated block (a whole repo) sent before `prompt` and
        marked for prompt caching, so later questions about the same repo read it cheaply.
        """
        ...


class TokenCounter(Protocol):
    def count_tokens(self, system: str, user_texts: list[str]) -> int:
        """Exact input tokens for a request with this system prompt and these user texts."""
        ...


class CostLog(Protocol):
    def record(self, entry: CostLogEntry) -> None: ...


class Answerer(Protocol):
    """RAG and long-context modes both implement this, so the evaluation
    harness can run the same question through either one."""

    def answer(self, repo_id: str, question: str) -> Answer: ...
