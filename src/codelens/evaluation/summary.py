"""Turn a results CSV into a table per mode and question type (markdown, for the paper)."""

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

ALL_TYPES = "all"


@dataclass(frozen=True)
class GroupSummary:
    mode: str
    question_type: str
    answers: int
    errors: int
    judge_score: float | None  # mean of 0-2 scores
    retrieval_recall: float | None
    cited_recall: float | None
    cost_per_answer_usd: float  # answer + judge
    total_cost_usd: float
    latency_seconds: float | None
    truncated: int


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def _floats(rows: list[dict[str, str]], column: str) -> list[float]:
    return [float(r[column]) for r in rows if r[column] not in ("", None)]


def _summarize(mode: str, question_type: str, rows: list[dict[str, str]]) -> GroupSummary:
    ok = [r for r in rows if not r["error"]]
    costs = [float(r["answer_cost_usd"]) + float(r["judge_cost_usd"]) for r in rows]
    return GroupSummary(
        mode=mode,
        question_type=question_type,
        answers=len(ok),
        errors=len(rows) - len(ok),
        judge_score=_mean(_floats(ok, "judge_score")),
        retrieval_recall=_mean(_floats(ok, "retrieval_recall")),
        cited_recall=_mean(_floats(ok, "cited_recall")),
        cost_per_answer_usd=sum(costs) / len(rows) if rows else 0.0,
        total_cost_usd=sum(costs),
        latency_seconds=_mean(_floats(ok, "latency_seconds")),
        truncated=sum(r["context_truncated"] == "True" for r in ok),
    )


def summarize(path: Path) -> list[GroupSummary]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    groups: list[GroupSummary] = []
    for mode in sorted({r["mode"] for r in rows}):
        mode_rows = [r for r in rows if r["mode"] == mode]
        for question_type in sorted({r["question_type"] for r in mode_rows}):
            type_rows = [r for r in mode_rows if r["question_type"] == question_type]
            groups.append(_summarize(mode, question_type, type_rows))
        groups.append(_summarize(mode, ALL_TYPES, mode_rows))
    return groups


def _fmt(value: float | None, digits: int = 2) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def to_markdown(groups: list[GroupSummary]) -> str:
    header = (
        "| mode | question type | answers | errors | judge (0-2) | retrieval recall "
        "| cited recall | $ per answer | $ total | seconds | truncated |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    lines = [
        f"| {g.mode} | {g.question_type} | {g.answers} | {g.errors} | {_fmt(g.judge_score)} "
        f"| {_fmt(g.retrieval_recall)} | {_fmt(g.cited_recall)} | {g.cost_per_answer_usd:.4f} "
        f"| {g.total_cost_usd:.4f} | {_fmt(g.latency_seconds, 1)} | {g.truncated} |"
        for g in groups
    ]
    return "\n".join([header, *lines])
