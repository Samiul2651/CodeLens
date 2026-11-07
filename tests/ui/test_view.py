from codelens.api.schemas import AskResponse, CostResponse
from codelens.core.models import AnswerMode
from codelens.ui.view import SourceRow, cost_rows, notes, source_rows

COST = CostResponse(
    model="claude-opus-5-5",
    input_tokens=1_400,
    output_tokens=600,
    cache_write_tokens=0,
    cache_read_tokens=10_700,
    cost_usd=0.01234,
    latency_seconds=4.25,
)


def response(mode: AnswerMode, **changes: object) -> AskResponse:
    fields: dict[str, object] = {
        "answer": "text",
        "mode": mode,
        "sources": ["a.py:1-5", "b.py:2-9"],
        "cited": ["b.py:2-9"],
        "refused": False,
        "context_truncated": False,
        "cost": COST,
    }
    return AskResponse.model_validate(fields | changes)


def test_rag_lists_every_source_and_marks_the_cited_ones() -> None:
    assert source_rows(response(AnswerMode.RAG)) == [
        SourceRow("a.py:1-5", cited=False),
        SourceRow("b.py:2-9", cited=True),
    ]


def test_long_context_lists_only_cited_sources() -> None:
    assert source_rows(response(AnswerMode.LONG_CONTEXT)) == [SourceRow("b.py:2-9", cited=True)]


def test_notes_for_refusal_and_truncation() -> None:
    assert notes(response(AnswerMode.RAG)) == []
    found = notes(response(AnswerMode.LONG_CONTEXT, refused=True, context_truncated=True))
    assert len(found) == 2
    assert "declined" in found[0]
    assert "token budget" in found[1]


def test_cost_rows_are_formatted() -> None:
    rows = dict(cost_rows(COST))
    assert rows["Cost"] == "$0.0123"
    assert rows["Cache read tokens"] == "10,700"
    assert rows["Time"] == "4.2 s"
