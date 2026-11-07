"""Scores that need no model: did the right files reach the model, and did the answer cite them?

retrieval_recall  share of expected files among the chunks SHOWN to the model.
                  RAG: did search find them? Long-context: did they fit the budget?
cited_recall      share of expected files the answer CITES.
cited_precision   share of cited files that are expected (low = cites unrelated code).

None means "not defined" (no expected files, or nothing cited), not zero.
"""


def file_of(source: str) -> str:
    """'app/auth.py:15-17' -> 'app/auth.py'"""
    return source.rsplit(":", 1)[0]


def _files(sources: list[str]) -> set[str]:
    return {file_of(s) for s in sources}


def recall(expected_files: list[str], sources: list[str]) -> float | None:
    if not expected_files:
        return None
    found = _files(sources)
    return sum(f in found for f in expected_files) / len(expected_files)


def precision(expected_files: list[str], sources: list[str]) -> float | None:
    cited = _files(sources)
    if not expected_files or not cited:
        return None
    return len(cited & set(expected_files)) / len(cited)
