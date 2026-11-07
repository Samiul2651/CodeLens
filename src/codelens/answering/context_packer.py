"""Choose which files go into a long-context prompt when the whole repo does not fit.

Rules:
- Whole files only: a file is in or out, never cut in half.
- Most useful files first (PRIORITY below), then by path, so the result is deterministic.
- Plan with a cheap estimate (characters / CHARS_PER_TOKEN), then check the result with ONE
  exact token count. If the estimate was too low, drop files from the end and check again.
  (Counting every file separately would be one API call per file.)
"""

from dataclasses import dataclass
from itertools import groupby
from pathlib import PurePosixPath

from codelens.answering.prompts import SYSTEM_PROMPT, format_excerpts, question_block
from codelens.core.interfaces import TokenCounter
from codelens.core.models import Chunk

CHARS_PER_TOKEN = 3.0  # code is token-dense; a low value keeps the first estimate safe
MAX_SHRINK_ROUNDS = 5
TEST_MARKERS = ("test", "tests", "spec", "__tests__")


@dataclass(frozen=True)
class PackedContext:
    context: str  # the <excerpts> block, "" if nothing fits
    chunks: list[Chunk]
    files_included: int
    files_total: int
    input_tokens: int  # exact count of the full request (system + context + question)

    @property
    def truncated(self) -> bool:
        return self.files_included < self.files_total


def file_priority(path: str) -> int:
    """0 = read first. Root docs explain the repo, then source, then tests, then config."""
    pure = PurePosixPath(path)
    parts = [p.lower() for p in pure.parts]
    if len(parts) == 1 and pure.stem.lower() in {"readme", "architecture", "contributing"}:
        return 0
    if any(p in TEST_MARKERS or p.startswith("test_") for p in parts) or pure.stem.endswith(
        "_test"
    ):
        return 2
    if pure.suffix.lower() in {".py", ".js", ".java", ".cs"}:
        return 1
    return 3


def _context_text(chunks: list[Chunk]) -> str:
    return f"<excerpts>\n{format_excerpts(chunks)}\n</excerpts>" if chunks else ""


class ContextPacker:
    def __init__(self, counter: TokenCounter, token_budget: int) -> None:
        if token_budget < 1:
            raise ValueError("token_budget must be >= 1")
        self._counter = counter
        self._budget = token_budget

    def pack(self, chunks: list[Chunk], question: str) -> PackedContext:
        files = [
            list(file_chunks)
            for _, file_chunks in groupby(
                sorted(chunks, key=lambda c: (c.file_path, c.start_line)), key=lambda c: c.file_path
            )
        ]
        files.sort(key=lambda fc: (file_priority(fc[0].file_path), fc[0].file_path))
        prompt = question_block(question)
        fixed = (len(SYSTEM_PROMPT) + len(prompt)) / CHARS_PER_TOKEN

        # 1) Estimate: take files in priority order while they fit.
        chosen: list[list[Chunk]] = []
        estimate = fixed
        for file_chunks in files:
            size = len(_context_text(file_chunks)) / CHARS_PER_TOKEN
            if estimate + size > self._budget:
                break
            chosen.append(file_chunks)
            estimate += size

        # 2) Verify with one exact count; shrink if the estimate was optimistic.
        for _ in range(MAX_SHRINK_ROUNDS):
            selected = [c for fc in chosen for c in fc]
            context = _context_text(selected)
            tokens = self._counter.count_tokens(SYSTEM_PROMPT, [t for t in (context, prompt) if t])
            if tokens <= self._budget or not chosen:
                return PackedContext(context, selected, len(chosen), len(files), tokens)
            keep = max(0, int(len(chosen) * self._budget / tokens) - 1)
            chosen = chosen[:keep]
        return PackedContext("", [], 0, len(files), tokens)
