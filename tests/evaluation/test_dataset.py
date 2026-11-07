import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from codelens.evaluation.dataset import QuestionType, load_question_set

PILOT = Path(__file__).parents[2] / "datasets" / "fastapi-basics.json"


def write(tmp_path: Path, questions: list[dict[str, object]]) -> Path:
    path = tmp_path / "set.json"
    path.write_text(json.dumps({"name": "t", "questions": questions}), encoding="utf-8")
    return path


def question(**overrides: object) -> dict[str, object]:
    return {
        "id": "q1",
        "repo_id": "demo",
        "question": "Where?",
        "question_type": "locate",
        "expected_answer": "Here.",
        "expected_files": ["./app\\auth.py"],
        **overrides,
    }


def test_pilot_set_loads() -> None:
    question_set = load_question_set(PILOT)

    assert len(question_set.questions) == 9
    assert {q.question_type for q in question_set.questions} == set(QuestionType)
    assert all(q.expected_files for q in question_set.questions)


def test_paths_are_normalized(tmp_path: Path) -> None:
    [q] = load_question_set(write(tmp_path, [question()])).questions

    assert q.expected_files == ["app/auth.py"]


@pytest.mark.parametrize(
    "questions",
    [
        [question(), question()],  # duplicate id
        [question(repo_id="../x")],
        [question(question_type="trivia")],
        [question(expected_answer="")],
        [],
    ],
)
def test_invalid_sets_are_rejected(tmp_path: Path, questions: list[dict[str, object]]) -> None:
    with pytest.raises(ValidationError):
        load_question_set(write(tmp_path, questions))
