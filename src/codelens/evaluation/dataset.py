"""Question sets for the evaluation, stored as JSON.

{
  "name": "fastapi-basics pilot",
  "questions": [
    {
      "id": "fb-01",
      "repo_id": "fastapi-basics",
      "question": "How are snippet ids assigned?",
      "question_type": "locate",
      "expected_answer": "InMemorySnippetRepository.add uses a counter starting at 1 ...",
      "expected_files": ["src/fastapi_basics/tasks/task3_repository.py"]
    }
  ]
}

question_type (for RQ1, "does it depend on the question type?"):
  locate        find where one thing is done (one function or file)
  explain_flow  follow behaviour across several files
  architecture  the big picture: structure, layers, design choices
"""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, Field, field_validator, model_validator

REPO_ID_PATTERN = r"^[A-Za-z0-9._-]{1,100}$"


class QuestionType(StrEnum):
    LOCATE = "locate"
    EXPLAIN_FLOW = "explain_flow"
    ARCHITECTURE = "architecture"


class EvalQuestion(BaseModel):
    id: str = Field(min_length=1, max_length=50)
    repo_id: str = Field(pattern=REPO_ID_PATTERN)
    question: str = Field(min_length=1, max_length=2_000)
    question_type: QuestionType
    expected_answer: str = Field(min_length=1)
    expected_files: list[str] = Field(default_factory=list)  # repo-relative, "/" separators

    @field_validator("expected_files")
    @classmethod
    def normalize_paths(cls, paths: list[str]) -> list[str]:
        return [p.replace("\\", "/").lstrip("./") for p in paths]


class QuestionSet(BaseModel):
    name: str
    questions: list[EvalQuestion] = Field(min_length=1)

    @model_validator(mode="after")
    def ids_are_unique(self) -> "QuestionSet":
        ids = [q.id for q in self.questions]
        duplicates = sorted({i for i in ids if ids.count(i) > 1})
        if duplicates:
            raise ValueError(f"duplicate question ids: {duplicates}")
        return self


def load_question_set(path: Path) -> QuestionSet:
    return QuestionSet.model_validate_json(path.read_text(encoding="utf-8"))
