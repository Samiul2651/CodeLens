"""Run a question set through the answer modes and write one CSV row per answer.

- Rows are written as soon as each answer is scored, so a crash keeps what was done.
- Re-running with the same output file skips (question, mode, run) rows already there,
  so an interrupted run can be resumed.
- A cost cap stops the run before it overspends (the paper budget is about $50).
- One question failing (network, refusal...) is recorded in the `error` column; the run goes on.
- Order: repo -> mode -> question. Long-context questions about one repo run back to back,
  so the cached repo block is reused (cache lifetime is 5 minutes).
"""

import csv
import logging
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from codelens.core.interfaces import Answerer
from codelens.core.models import AnswerMode
from codelens.evaluation.dataset import EvalQuestion, QuestionSet
from codelens.evaluation.judge import Judge
from codelens.evaluation.scoring import precision, recall

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalRow:
    question_id: str
    repo_id: str
    question_type: str
    mode: str
    run: int
    answer: str = ""
    sources_shown: int = 0
    retrieval_recall: float | None = None
    cited_recall: float | None = None
    cited_precision: float | None = None
    judge_score: int | None = None
    judge_reason: str = ""
    context_truncated: bool = False
    refused: bool = False
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    answer_cost_usd: float = 0.0
    judge_cost_usd: float = 0.0
    latency_seconds: float = 0.0
    error: str = ""

    @property
    def total_cost_usd(self) -> float:
        return self.answer_cost_usd + self.judge_cost_usd


COLUMNS = [f.name for f in fields(EvalRow)]


@dataclass(frozen=True)
class RunReport:
    written: int
    skipped: int
    failed: int
    cost_usd: float
    stopped_by_cost_cap: bool


def _key(question_id: str, mode: str, run: int) -> tuple[str, str, int]:
    return question_id, mode, run


def _done_keys(path: Path) -> set[tuple[str, str, int]]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as f:
        return {_key(r["question_id"], r["mode"], int(r["run"])) for r in csv.DictReader(f)}


class EvalRunner:
    def __init__(
        self,
        answerers: dict[AnswerMode, Answerer],
        judge: Judge | None,
        on_row: Callable[[EvalRow], None] = lambda row: None,
    ) -> None:
        self._answerers = answerers
        self._judge = judge
        self._on_row = on_row

    def run(
        self,
        question_set: QuestionSet,
        modes: list[AnswerMode],
        out_path: Path,
        runs: int = 1,
        max_cost_usd: float | None = None,
    ) -> RunReport:
        missing = [m for m in modes if m not in self._answerers]
        if missing:
            raise ValueError(f"no answerer for modes: {missing}")
        done = _done_keys(out_path)
        written = skipped = failed = 0
        spent = 0.0

        for question, mode, run in self._plan(question_set, modes, runs):
            if _key(question.id, mode, run) in done:
                skipped += 1
                continue
            if max_cost_usd is not None and spent >= max_cost_usd:
                return RunReport(written, skipped, failed, spent, stopped_by_cost_cap=True)
            row = self._evaluate(question, mode, run)
            self._append(out_path, row)
            self._on_row(row)
            written += 1
            failed += bool(row.error)
            spent += row.total_cost_usd
        return RunReport(written, skipped, failed, spent, stopped_by_cost_cap=False)

    @staticmethod
    def _plan(
        question_set: QuestionSet, modes: list[AnswerMode], runs: int
    ) -> list[tuple[EvalQuestion, AnswerMode, int]]:
        repos = list(dict.fromkeys(q.repo_id for q in question_set.questions))
        return [
            (question, mode, run)
            for repo in repos
            for mode in modes
            for run in range(1, runs + 1)
            for question in question_set.questions
            if question.repo_id == repo
        ]

    def _evaluate(self, question: EvalQuestion, mode: AnswerMode, run: int) -> EvalRow:
        base = EvalRow(question.id, question.repo_id, question.question_type, mode, run)
        try:
            answer = self._answerers[mode].answer(question.repo_id, question.question)
        except Exception as error:  # one bad question must not stop a long, paid run
            logger.exception("answer failed for %s (%s)", question.id, mode)
            return EvalRow(**{**asdict(base), "error": f"{type(error).__name__}: {error}"})

        verdict = None
        if self._judge is not None and not answer.refused and answer.cost is not None:
            verdict = self._judge.judge(question.question, question.expected_answer, answer.text)
        cost = answer.cost
        return EvalRow(
            **{
                **asdict(base),
                "answer": answer.text,
                "sources_shown": len(answer.sources),
                "retrieval_recall": recall(question.expected_files, answer.sources),
                "cited_recall": recall(question.expected_files, answer.cited),
                "cited_precision": precision(question.expected_files, answer.cited),
                "judge_score": verdict.score if verdict else None,
                "judge_reason": verdict.reason if verdict else "",
                "context_truncated": answer.context_truncated,
                "refused": answer.refused,
                "model": cost.model if cost else "",
                "input_tokens": cost.input_tokens if cost else 0,
                "output_tokens": cost.output_tokens if cost else 0,
                "cache_write_tokens": cost.cache_write_tokens if cost else 0,
                "cache_read_tokens": cost.cache_read_tokens if cost else 0,
                "answer_cost_usd": cost.cost_usd if cost else 0.0,
                "judge_cost_usd": verdict.cost.cost_usd if verdict and verdict.cost else 0.0,
                "latency_seconds": cost.latency_seconds if cost else 0.0,
            }
        )

    @staticmethod
    def _append(path: Path, row: EvalRow) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        is_new = not path.exists()
        with path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            if is_new:
                writer.writeheader()
            writer.writerow({k: "" if v is None else v for k, v in asdict(row).items()})
