"""Test doubles for the model-backed parts."""

import re
import zlib

from codelens.core.models import Completion, CostLogEntry, CostRecord

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


class FakeLanguageModel:
    """Answers with a fixed text and records every prompt."""

    def __init__(self, text: str = "Answer.", refused: bool = False) -> None:
        self.calls: list[tuple[str, str]] = []
        self.cached_contexts: list[str | None] = []
        self._text = text
        self._refused = refused

    def complete(self, system: str, prompt: str, cached_context: str | None = None) -> Completion:
        self.calls.append((system, prompt))
        self.cached_contexts.append(cached_context)
        cost = CostRecord("claude-opus-5-5", 1000, 100, cost_usd=0.006, latency_seconds=1.5)
        if self._refused:
            return Completion("", cost, refused=True, refusal_category="cyber")
        return Completion(self._text, cost)


class CharTokenCounter:
    """Fake exact counter: `chars_per_token` characters = 1 token. Records each call."""

    def __init__(self, chars_per_token: float = 3.0) -> None:
        self.calls = 0
        self._chars_per_token = chars_per_token

    def count_tokens(self, system: str, user_texts: list[str]) -> int:
        self.calls += 1
        return int((len(system) + sum(len(t) for t in user_texts)) / self._chars_per_token)


class ListCostLog:
    def __init__(self) -> None:
        self.entries: list[CostLogEntry] = []

    def record(self, entry: CostLogEntry) -> None:
        self.entries.append(entry)
