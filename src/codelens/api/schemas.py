"""Request and response models for the HTTP API."""

from pydantic import BaseModel, Field

from codelens.core.models import Answer, AnswerMode

REPO_ID_PATTERN = r"^[A-Za-z0-9._-]{1,100}$"


class AskRequest(BaseModel):
    repo_id: str = Field(pattern=REPO_ID_PATTERN)
    question: str = Field(min_length=1, max_length=2_000)
    mode: AnswerMode = AnswerMode.RAG


class CostResponse(BaseModel):
    model: str
    input_tokens: int
    output_tokens: int
    cache_write_tokens: int
    cache_read_tokens: int
    cost_usd: float
    latency_seconds: float


class AskResponse(BaseModel):
    answer: str
    mode: AnswerMode
    sources: list[str]
    cited: list[str]
    refused: bool
    context_truncated: bool  # long-context mode: some files did not fit the budget
    cost: CostResponse | None

    @classmethod
    def from_answer(cls, answer: Answer) -> "AskResponse":
        cost = answer.cost
        return cls(
            answer=answer.text,
            mode=answer.mode,
            sources=answer.sources,
            cited=answer.cited,
            refused=answer.refused,
            context_truncated=answer.context_truncated,
            cost=CostResponse(**vars(cost)) if cost is not None else None,
        )
