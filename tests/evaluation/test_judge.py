import pytest

from codelens.evaluation.judge import JUDGE_SYSTEM_PROMPT, LLMJudge, parse_verdict
from tests.fakes import FakeLanguageModel


@pytest.mark.parametrize(
    ("text", "score"),
    [
        ('{"score": 2, "reason": "matches"}', 2),
        ('```json\n{"score": 1, "reason": "misses the lock"}\n```', 1),
        ('Here you go: {"score": 0, "reason": "wrong file"}', 0),
    ],
)
def test_parse_valid_replies(text: str, score: int) -> None:
    assert parse_verdict(text)[0] == score


@pytest.mark.parametrize(
    "text", ["no json here", '{"score": 5}', '{"score": "2"}', '{"score": true}', "{broken"]
)
def test_unreadable_replies_give_none(text: str) -> None:
    score, reason = parse_verdict(text)

    assert score is None
    assert reason


def test_judge_sends_question_reference_and_candidate() -> None:
    model = FakeLanguageModel('{"score": 2, "reason": "same facts"}')

    verdict = LLMJudge(model).judge("Where?", "In a.py.", "It is in a.py [a.py:1-2].")

    [(system, prompt)] = model.calls
    assert system == JUDGE_SYSTEM_PROMPT
    assert "<question>\nWhere?\n</question>" in prompt
    assert "<reference_answer>\nIn a.py.\n</reference_answer>" in prompt
    assert "<candidate_answer>\nIt is in a.py [a.py:1-2].\n</candidate_answer>" in prompt
    assert (verdict.score, verdict.reason) == (2, "same facts")
    assert verdict.cost is not None and verdict.cost.cost_usd == pytest.approx(0.006)


def test_judge_refusal() -> None:
    verdict = LLMJudge(FakeLanguageModel(refused=True)).judge("q", "r", "c")

    assert verdict.score is None
