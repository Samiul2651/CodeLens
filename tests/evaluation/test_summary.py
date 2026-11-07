from pathlib import Path

import pytest

from codelens.core.models import Answer, AnswerMode, CostRecord
from codelens.evaluation.dataset import EvalQuestion, QuestionSet, QuestionType
from codelens.evaluation.judge import Verdict
from codelens.evaluation.runner import EvalRunner
from codelens.evaluation.summary import summarize, to_markdown


class FixedAnswerer:
    def __init__(self, cost: float, cited: list[str]) -> None:
        self._cost, self._cited = cost, cited

    def answer(self, repo_id: str, question: str) -> Answer:
        return Answer(
            "answer", AnswerMode.RAG, sources=["a.py:1-2"], cited=self._cited,
            cost=CostRecord("claude-opus-5-5", 10, 10, cost_usd=self._cost, latency_seconds=3.0),
        )  # fmt: skip


class ScoreByQuestion:
    def judge(self, question: str, reference: str, candidate: str) -> Verdict:
        return Verdict(2 if "locate" in question else 0, "", None)


def test_summary_per_mode_and_type(tmp_path: Path) -> None:
    questions = QuestionSet(
        name="t",
        questions=[
            EvalQuestion(id=f"q{i}", repo_id="demo", question=f"{t.value} {i}", question_type=t,
                         expected_answer="r", expected_files=["a.py"])
            for i, t in enumerate([QuestionType.LOCATE, QuestionType.LOCATE,
                                   QuestionType.ARCHITECTURE])
        ],
    )  # fmt: skip
    out = tmp_path / "r.csv"
    answerers = {
        AnswerMode.RAG: FixedAnswerer(0.01, ["a.py:1-2"]),
        AnswerMode.LONG_CONTEXT: FixedAnswerer(0.05, []),
    }
    EvalRunner(answerers, ScoreByQuestion()).run(  # type: ignore[arg-type]
        questions, [AnswerMode.RAG, AnswerMode.LONG_CONTEXT], out
    )

    groups = {(g.mode, g.question_type): g for g in summarize(out)}

    rag_locate = groups[("rag", "locate")]
    assert rag_locate.answers == 2
    assert rag_locate.judge_score == 2.0
    assert rag_locate.cited_recall == 1.0
    assert groups[("rag", "architecture")].judge_score == 0.0
    assert groups[("rag", "all")].judge_score == pytest.approx(4 / 3)
    assert groups[("long_context", "all")].cited_recall == 0.0
    assert groups[("long_context", "all")].total_cost_usd == pytest.approx(0.15)
    table = to_markdown(summarize(out))
    assert table.startswith("| mode | question type |")
    assert "| rag | all | 3 | 0 | 1.33 |" in table
