import csv
from pathlib import Path

import pytest

from codelens.core.models import Answer, AnswerMode, CostRecord
from codelens.evaluation.dataset import EvalQuestion, QuestionSet, QuestionType
from codelens.evaluation.judge import Verdict
from codelens.evaluation.runner import COLUMNS, EvalRunner


class StubAnswerer:
    def __init__(self, mode: AnswerMode, cost: float = 0.01, fail_on: str = "") -> None:
        self.asked: list[tuple[str, str]] = []
        self._mode = mode
        self._cost = cost
        self._fail_on = fail_on

    def answer(self, repo_id: str, question: str) -> Answer:
        self.asked.append((repo_id, question))
        if question == self._fail_on:
            raise ConnectionError("Qdrant is down")
        return Answer(
            text=f"{self._mode} answer [a.py:1-5]",
            mode=self._mode,
            sources=["a.py:1-5", "b.py:1-3"],
            cited=["a.py:1-5"],
            cost=CostRecord("claude-opus-5-5", 1000, 100, 50, 25, self._cost, 2.0),
            context_truncated=self._mode is AnswerMode.LONG_CONTEXT,
        )


class StubJudge:
    def __init__(self) -> None:
        self.calls = 0

    def judge(self, question: str, reference: str, candidate: str) -> Verdict:
        self.calls += 1
        return Verdict(2, "ok", CostRecord("claude-opus-5-5", 300, 30, cost_usd=0.002))


def q(qid: str, repo: str = "demo") -> EvalQuestion:
    return EvalQuestion(
        id=qid, repo_id=repo, question=f"question {qid}", question_type=QuestionType.LOCATE,
        expected_answer="ref", expected_files=["a.py", "c.py"],
    )  # fmt: skip


QUESTIONS = QuestionSet(name="t", questions=[q("q1"), q("q2"), q("x1", repo="other")])
MODES = [AnswerMode.RAG, AnswerMode.LONG_CONTEXT]


def make_runner(rag_cost: float = 0.01, fail_on: str = "") -> tuple[EvalRunner, StubJudge]:
    judge = StubJudge()
    answerers = {
        AnswerMode.RAG: StubAnswerer(AnswerMode.RAG, rag_cost, fail_on),
        AnswerMode.LONG_CONTEXT: StubAnswerer(AnswerMode.LONG_CONTEXT),
    }
    return EvalRunner(answerers, judge), judge  # type: ignore[arg-type]


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_writes_one_scored_row_per_question_and_mode(tmp_path: Path) -> None:
    runner, judge = make_runner()
    out = tmp_path / "results.csv"

    report = runner.run(QUESTIONS, MODES, out)

    rows = read(out)
    assert list(rows[0]) == COLUMNS
    assert len(rows) == report.written == 6
    assert judge.calls == 6
    first = rows[0]
    assert (first["question_id"], first["mode"], first["run"]) == ("q1", "rag", "1")
    assert first["retrieval_recall"] == "0.5"  # a.py shown, c.py not
    assert first["cited_recall"] == "0.5"
    assert first["cited_precision"] == "1.0"
    assert first["judge_score"] == "2"
    assert float(first["answer_cost_usd"]) == pytest.approx(0.01)
    assert float(first["judge_cost_usd"]) == pytest.approx(0.002)
    assert report.cost_usd == pytest.approx(6 * 0.012)


def test_order_keeps_each_repo_and_mode_together(tmp_path: Path) -> None:
    runner, _ = make_runner()
    out = tmp_path / "results.csv"

    runner.run(QUESTIONS, MODES, out, runs=2)

    order = [(r["repo_id"], r["mode"]) for r in read(out)]
    assert order == (
        [("demo", "rag")] * 4 + [("demo", "long_context")] * 4
        + [("other", "rag")] * 2 + [("other", "long_context")] * 2
    )  # fmt: skip


def test_resume_skips_rows_already_written(tmp_path: Path) -> None:
    out = tmp_path / "results.csv"
    make_runner()[0].run(QUESTIONS, [AnswerMode.RAG], out)

    runner, judge = make_runner()
    report = runner.run(QUESTIONS, MODES, out)

    assert (report.skipped, report.written) == (3, 3)
    assert judge.calls == 3
    assert len(read(out)) == 6


def test_cost_cap_stops_the_run(tmp_path: Path) -> None:
    runner, _ = make_runner(rag_cost=0.50)

    report = runner.run(QUESTIONS, MODES, tmp_path / "results.csv", max_cost_usd=1.0)

    assert report.stopped_by_cost_cap
    assert report.written == 2  # 2 x ($0.50 + $0.002) >= $1.00
    assert report.cost_usd >= 1.0


def test_a_failing_question_is_recorded_and_the_run_goes_on(tmp_path: Path) -> None:
    runner, _ = make_runner(fail_on="question q1")
    out = tmp_path / "results.csv"

    report = runner.run(QUESTIONS, [AnswerMode.RAG], out)

    rows = read(out)
    assert report.failed == 1 and report.written == 3
    assert rows[0]["error"] == "ConnectionError: Qdrant is down"
    assert rows[0]["judge_score"] == ""
    assert rows[1]["error"] == ""


def test_unknown_mode_is_rejected(tmp_path: Path) -> None:
    runner = EvalRunner({AnswerMode.RAG: StubAnswerer(AnswerMode.RAG)}, None)

    with pytest.raises(ValueError):
        runner.run(QUESTIONS, MODES, tmp_path / "r.csv")


def test_without_judge_scores_are_empty(tmp_path: Path) -> None:
    runner = EvalRunner({AnswerMode.RAG: StubAnswerer(AnswerMode.RAG)}, judge=None)
    out = tmp_path / "r.csv"

    runner.run(QUESTIONS, [AnswerMode.RAG], out)

    assert {r["judge_score"] for r in read(out)} == {""}
