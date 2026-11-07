"""Chunks and their vectors in one Qdrant collection, filtered by repo_id."""

import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    FilterSelector,
    MatchAny,
    MatchValue,
    PayloadSchemaType,
    PointStruct,
    VectorParams,
)

from codelens.core.models import Chunk, RetrievedChunk


def point_id(chunk: Chunk) -> str:
    """Stable id: re-indexing the same chunk replaces it instead of adding a copy."""
    key = f"{chunk.repo_id}:{chunk.file_path}:{chunk.start_line}"
    return str(uuid.uuid5(uuid.NAMESPACE_URL, key))


def _repo_filter(repo_id: str) -> FieldCondition:
    return FieldCondition(key="repo_id", match=MatchValue(value=repo_id))


class QdrantVectorStore:
    def __init__(self, client: QdrantClient, collection: str) -> None:
        self._client = client
        self._collection = collection

    def ensure_collection(self, vector_size: int) -> None:
        """Create the collection if needed. Refuse a different vector size (another model)."""
        if self._client.collection_exists(self._collection):
            vectors = self._client.get_collection(self._collection).config.params.vectors
            existing = (
                vectors.size if vectors is not None and not isinstance(vectors, dict) else None
            )
            if existing != vector_size:
                raise ValueError(
                    f"collection {self._collection!r} has vector size {existing}, "
                    f"but the embedder returns {vector_size}; use another collection"
                )
            return
        self._client.create_collection(
            collection_name=self._collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
        )
        for field in ("repo_id", "file_path"):
            self._client.create_payload_index(
                self._collection, field_name=field, field_schema=PayloadSchemaType.KEYWORD
            )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        points = [
            PointStruct(
                id=point_id(chunk),
                vector=vector,
                payload={
                    "repo_id": chunk.repo_id,
                    "file_path": chunk.file_path,
                    "language": chunk.language,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "symbol": chunk.symbol,
                    "text": chunk.text,
                },
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self._client.upsert(collection_name=self._collection, points=points)

    def delete_files(self, repo_id: str, file_paths: list[str]) -> None:
        if not file_paths or not self._client.collection_exists(self._collection):
            return
        selector = FilterSelector(
            filter=Filter(
                must=[
                    _repo_filter(repo_id),
                    FieldCondition(key="file_path", match=MatchAny(any=file_paths)),
                ]
            )
        )
        self._client.delete(collection_name=self._collection, points_selector=selector)

    def search(self, repo_id: str, vector: list[float], limit: int) -> list[RetrievedChunk]:
        if not self._client.collection_exists(self._collection):
            return []  # nothing ingested yet
        response = self._client.query_points(
            collection_name=self._collection,
            query=vector,
            query_filter=Filter(must=[_repo_filter(repo_id)]),
            limit=limit,
            with_payload=True,
        )
        return [
            RetrievedChunk(chunk=_chunk_from_payload(point.payload or {}), score=point.score)
            for point in response.points
        ]

    def count(self, repo_id: str) -> int:
        if not self._client.collection_exists(self._collection):
            return 0
        return self._client.count(
            self._collection, count_filter=Filter(must=[_repo_filter(repo_id)]), exact=True
        ).count


def _chunk_from_payload(payload: dict[str, object]) -> Chunk:
    symbol = payload.get("symbol")
    return Chunk(
        repo_id=str(payload["repo_id"]),
        file_path=str(payload["file_path"]),
        language=str(payload["language"]),
        start_line=int(str(payload["start_line"])),
        end_line=int(str(payload["end_line"])),
        text=str(payload["text"]),
        symbol=str(symbol) if symbol is not None else None,
    )
