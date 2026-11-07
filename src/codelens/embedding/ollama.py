"""Embeddings from a local Ollama server."""

import httpx

EMBED_TIMEOUT_SECONDS = 300.0


class OllamaEmbedder:
    def __init__(self, base_url: str, model: str, http: httpx.Client | None = None) -> None:
        self._url = f"{base_url.rstrip('/')}/api/embed"
        self._model = model
        self._http = http or httpx.Client(timeout=EMBED_TIMEOUT_SECONDS)

    def embed(self, texts: list[str]) -> list[list[float]]:
        """One request for the whole list. The ingestion service batches large inputs."""
        if not texts:
            return []
        response = self._http.post(self._url, json={"model": self._model, "input": texts})
        response.raise_for_status()
        embeddings: list[list[float]] = response.json()["embeddings"]
        if len(embeddings) != len(texts):
            raise RuntimeError(f"Ollama returned {len(embeddings)} vectors for {len(texts)} texts")
        return embeddings
