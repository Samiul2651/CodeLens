"""LLM judge: compares an answer with the reference answer and gives 0, 1 or 2.

The judge sees the question, the reference answer and the candidate answer, but not the
code and not which mode produced the answer, so it cannot favour one mode.
Judges make mistakes: hand-check a random sample and report how often you agree.
"""

import json
import re
from dataclasses import dataclass
from typing import Protocol

from codelens.core.interfaces import LanguageModel
from codelens.core.models import CostRecord

JUDGE_SYSTEM_PROMPT = (
    "You grade answers to questions about a code repository.\n"
    "Compare the candidate answer with the reference answer written by an expert.\n"
    "Score 2: correct and covers the key points of the reference.\n"
    "Score 1: partly correct, or correct but missing a key point.\n"
    "Score 0: wrong, mostly missing, or says it cannot answer.\n"
    "Extra correct detail is fine. Ignore style and length.\n"
    'Reply with only a JSON object: {"score": 0, 1 or 2, "reason": "one short sentence"}'
)

_JSON_OBJECT = re.compile(r"\{.*\}", re.DOTALL)


@dataclass(frozen=True)
class Verdict:
    score: int | None  # None if the judge's reply could not be read
    reason: str
    cost: CostRecord | None = None


class Judge(Protocol):
    def judge(self, question: str, reference: str, candidate: str) -> Verdict: ...


def parse_verdict(text: str) -> tuple[int | None, str]:
    match = _JSON_OBJECT.search(text)
    if match is None:
        return None, f"unreadable judge reply: {text[:200]}"
    try:
        data = json.loads(match.group(0))
        score = data["score"]
        reason = str(data.get("reason", ""))
    except (json.JSONDecodeError, KeyError, TypeError):
        return None, f"unreadable judge reply: {text[:200]}"
    if score not in (0, 1, 2) or isinstance(score, bool):
        return None, f"judge score out of range: {score!r}"
    return int(score), reason


class LLMJudge:
    def __init__(self, model: LanguageModel) -> None:
        self._model = model

    def judge(self, question: str, reference: str, candidate: str) -> Verdict:
        prompt = (
            f"<question>\n{question}\n</question>\n\n"
            f"<reference_answer>\n{reference}\n</reference_answer>\n\n"
            f"<candidate_answer>\n{candidate}\n</candidate_answer>"
        )
        completion = self._model.complete(JUDGE_SYSTEM_PROMPT, prompt)
        if completion.refused:
            return Verdict(None, "judge refused", completion.cost)
        score, reason = parse_verdict(completion.text)
        return Verdict(score, reason, completion.cost)
