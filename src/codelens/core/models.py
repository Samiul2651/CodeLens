"""Data shared across CodeLens. Plain dataclasses: no Qdrant, Ollama or Anthropic types.

Line numbers are 1-based and inclusive, as an editor shows them.
"""

from dataclasses import dataclass, field
from enum import StrEnum


class AnswerMode(StrEnum):
    RAG = "rag"
    LONG_CONTEXT = "long_context"


@dataclass(frozen=True)
class Chunk:
    """A piece of a file, ready to embed."""

    repo_id: str
    file_path: str  # relative, with "/" separators
    language: str
    start_line: int
    end_line: int
    text: str
    symbol: str | None = None  # e.g. "AuthService.login"

    @property
    def source(self) -> str:
        return f"{self.file_path}:{self.start_line}-{self.end_line}"


@dataclass(frozen=True)
class RetrievedChunk:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class CostRecord:
    """Tokens and dollars for one LLM call. Logged for every call (paper budget)."""

    model: str
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    cost_usd: float = 0.0
    latency_seconds: float = 0.0


@dataclass(frozen=True)
class Completion:
    """One language-model answer."""

    text: str
    cost: CostRecord
    refused: bool = False
    refusal_category: str | None = None


@dataclass(frozen=True)
class CostLogEntry:
    """One row of the cost log: which call, for what, and what it cost."""

    label: str  # e.g. "rag"
    repo_id: str
    question: str
    cost: CostRecord


@dataclass(frozen=True)
class Answer:
    text: str
    mode: AnswerMode
    sources: list[str] = field(default_factory=list)  # chunks shown to the model, best first
    cited: list[str] = field(default_factory=list)  # sources the answer actually cites
    cost: CostRecord | None = None
    refused: bool = False
    context_truncated: bool = False  # long-context mode: the repo did not fit
