from codelens.core.models import Answer, AnswerMode, Chunk


def test_chunk_source_label() -> None:
    chunk = Chunk("repo", "app/auth.py", "python", 15, 17, "def login(): ...", "AuthService.login")

    assert chunk.source == "app/auth.py:15-17"


def test_answer_defaults() -> None:
    answer = Answer(text="hi", mode=AnswerMode.RAG)

    assert answer.sources == []
    assert answer.cost is None
    assert answer.context_truncated is False
    assert AnswerMode("long_context") is AnswerMode.LONG_CONTEXT
