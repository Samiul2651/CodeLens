"""Test doubles for the model-backed parts."""

import re
import zlib

DIM = 64


class BagOfWordsEmbedder:
    """Deterministic fake: texts that share words get similar vectors."""

    def __init__(self) -> None:
        self.requests: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.requests.append(list(texts))
        return [self.vector(text) for text in texts]

    @staticmethod
    def vector(text: str) -> list[float]:
        vector = [0.0] * DIM
        vector[0] = 0.01  # never all zeros
        for word in re.findall(r"[a-z]+", text.lower()):
            vector[1 + zlib.crc32(word.encode()) % (DIM - 1)] += 1.0
        return vector


class InMemoryStateStore:
    def __init__(self) -> None:
        self.saved: dict[str, dict[str, str]] = {}

    def load(self, repo_id: str) -> dict[str, str]:
        return dict(self.saved.get(repo_id, {}))

    def save(self, repo_id: str, hashes: dict[str, str]) -> None:
        self.saved[repo_id] = dict(hashes)
