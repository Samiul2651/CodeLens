"""Two Chunker implementations, compared in the paper (RQ4):

- LineWindowChunker: fixed windows of lines, any file. The baseline.
- TreeSitterChunker: one chunk per function; leftover lines (imports, class headers)
  in line windows; plain-text files in line windows.

Both return chunks sorted by start line. Line numbers are 1-based and inclusive.
"""

from codelens.chunking.languages import LANGUAGE_BY_EXTENSION, PLAIN_TEXT_LANGUAGE, detect_language
from codelens.chunking.symbols import find_functions, outermost
from codelens.core.models import Chunk


def _windows(lines: list[str], window: int, overlap: int) -> list[tuple[int, int]]:
    """(start, end) line ranges, 1-based inclusive."""
    ranges: list[tuple[int, int]] = []
    start = 1
    while start <= len(lines):
        end = min(start + window - 1, len(lines))
        ranges.append((start, end))
        if end == len(lines):
            break
        start += window - overlap
    return ranges


def _make_chunks(
    repo_id: str,
    file_path: str,
    language: str,
    lines: list[str],
    first_line: int,
    last_line: int,
    window: int,
    symbol: str | None,
) -> list[Chunk]:
    """Split lines first_line..last_line into windows of at most `window` lines."""
    part = lines[first_line - 1 : last_line]
    shift = first_line - 1
    return [
        Chunk(
            repo_id=repo_id,
            file_path=file_path,
            language=language,
            start_line=start + shift,
            end_line=end + shift,
            text="\n".join(part[start - 1 : end]),
            symbol=symbol,
        )
        for start, end in _windows(part, window, overlap=0)
    ]


class LineWindowChunker:
    def __init__(self, window: int, overlap: int = 0) -> None:
        if window < 1 or overlap < 0 or overlap >= window:
            raise ValueError("need window >= 1 and 0 <= overlap < window")
        self._window = window
        self._overlap = overlap

    def chunk_file(self, repo_id: str, file_path: str, text: str) -> list[Chunk]:
        language = detect_language(file_path) or PLAIN_TEXT_LANGUAGE
        lines = text.splitlines()
        return [
            Chunk(repo_id, file_path, language, start, end, "\n".join(lines[start - 1 : end]))
            for start, end in _windows(lines, self._window, self._overlap)
        ]


class TreeSitterChunker:
    def __init__(self, max_lines: int) -> None:
        if max_lines < 1:
            raise ValueError("max_lines must be >= 1")
        self._max_lines = max_lines

    def chunk_file(self, repo_id: str, file_path: str, text: str) -> list[Chunk]:
        language = detect_language(file_path) or PLAIN_TEXT_LANGUAGE
        lines = text.splitlines()
        if language not in LANGUAGE_BY_EXTENSION.values():
            return _make_chunks(
                repo_id, file_path, language, lines, 1, len(lines), self._max_lines, None
            )

        chunks: list[Chunk] = []
        covered = [False] * (len(lines) + 1)
        for function in outermost(find_functions(text, language)):
            for line in range(function.start_line, function.end_line + 1):
                covered[line] = True
            chunks += _make_chunks(
                repo_id, file_path, language, lines,
                function.start_line, function.end_line, self._max_lines, function.name,
            )  # fmt: skip

        for first, last in _uncovered_runs(lines, covered):
            chunks += _make_chunks(
                repo_id, file_path, language, lines, first, last, self._max_lines, None
            )
        return sorted(chunks, key=lambda c: c.start_line)


def _has_content(lines: list[str]) -> bool:
    """False for runs like a lone "}" or "});" - noise for search."""
    return any(ch.isalnum() for line in lines for ch in line)


def _uncovered_runs(lines: list[str], covered: list[bool]) -> list[tuple[int, int]]:
    """Runs of consecutive lines outside any function, trimmed of blank lines at both ends,
    without runs that are only punctuation."""
    runs: list[tuple[int, int]] = []
    run: list[int] = []
    for line in range(1, len(lines) + 2):
        if line <= len(lines) and not covered[line]:
            run.append(line)
            continue
        while run and not lines[run[0] - 1].strip():
            run.pop(0)
        while run and not lines[run[-1] - 1].strip():
            run.pop()
        if run and _has_content(lines[run[0] - 1 : run[-1]]):
            runs.append((run[0], run[-1]))
        run = []
    return runs
