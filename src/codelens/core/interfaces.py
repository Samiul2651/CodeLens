"""The parts of CodeLens, as Protocols. Real versions live in their own packages;
tests use fakes. Everything else depends on these, never on a concrete class.
"""

from typing import Protocol

from codelens.core.models import Answer, Chunk, RetrievedChunk


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

    def delete_file(self, repo_id: str, file_path: str) -> None: ...

    def search(self, repo_id: str, vector: list[float], limit: int) -> list[RetrievedChunk]: ...


class Answerer(Protocol):
    """RAG and long-context modes both implement this, so the evaluation
    harness can run the same question through either one."""

    def answer(self, repo_id: str, question: str) -> Answer: ...
