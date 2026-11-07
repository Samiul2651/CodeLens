from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast

import pytest

from codelens.llm.claude import (
    FALLBACK_BETA,
    ClaudeLanguageModel,
    ClaudeTokenCounter,
    MissingCredentialsError,
    credentials_checked,
)


def make_message(
    text: str = "Paris.",
    stop_reason: str = "end_turn",
    model: str = "claude-opus-5-5",
    category: str | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking=""),
            SimpleNamespace(type="text", text=text),
        ],
        model=model,
        stop_reason=stop_reason,
        stop_details=SimpleNamespace(category=category) if stop_reason == "refusal" else None,
        usage=SimpleNamespace(
            input_tokens=1000,
            output_tokens=500,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
        ),
    )


class FakeAnthropic:
    def __init__(self, message: SimpleNamespace) -> None:
        self.calls: list[dict[str, Any]] = []
        self._message = message
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self._message


def ticking_clock(*times: float) -> Iterator[float]:
    yield from times


def make_model(client: FakeAnthropic) -> ClaudeLanguageModel:
    clock = ticking_clock(10.0, 12.5)
    return ClaudeLanguageModel(
        cast(Any, client), "claude-opus-5-5", "medium", 16_000, clock=lambda: next(clock)
    )


def test_sends_the_right_request() -> None:
    client = FakeAnthropic(make_message())

    make_model(client).complete("Be brief.", "Capital of France?")

    [call] = client.calls
    assert call["model"] == "claude-opus-5-5"
    assert call["system"] == "Be brief."
    assert call["messages"] == [{"role": "user", "content": "Capital of France?"}]
    assert call["output_config"] == {"effort": "medium"}
    assert call["betas"] == [FALLBACK_BETA]
    assert call["fallbacks"] == "default"
    assert call["max_tokens"] == 16_000


def test_returns_text_cost_and_latency() -> None:
    completion = make_model(FakeAnthropic(make_message())).complete("s", "p")

    assert completion.text == "Paris."
    assert not completion.refused
    assert completion.cost.input_tokens == 1000
    assert completion.cost.output_tokens == 500
    # 1000 x $4 / 1M + 500 x $20 / 1M = $0.014
    assert completion.cost.cost_usd == pytest.approx(0.014)
    assert completion.cost.latency_seconds == pytest.approx(2.5)


def test_cost_uses_the_model_that_answered() -> None:
    message = make_message(model="claude-sonnet-5-5")  # e.g. after a refusal fallback

    completion = make_model(FakeAnthropic(message)).complete("s", "p")

    # 1000 x $2 / 1M + 500 x $10 / 1M = $0.007
    assert completion.cost.model == "claude-sonnet-5-5"
    assert completion.cost.cost_usd == pytest.approx(0.007)


def test_refusal_is_reported_and_still_costed() -> None:
    message = make_message(stop_reason="refusal", category="cyber")

    completion = make_model(FakeAnthropic(message)).complete("s", "p")

    assert completion.refused
    assert completion.refusal_category == "cyber"
    assert completion.text == ""
    assert completion.cost.cost_usd > 0


def test_unknown_model_is_rejected_at_startup() -> None:
    with pytest.raises(ValueError, match="price"):
        ClaudeLanguageModel(cast(Any, object()), "claude-unknown", "low", 100)


def test_cached_context_goes_first_and_is_marked_for_caching() -> None:
    client = FakeAnthropic(make_message())

    make_model(client).complete("sys", "Question: q?", cached_context="<excerpts>repo</excerpts>")

    [call] = client.calls
    assert call["messages"] == [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<excerpts>repo</excerpts>",
                    "cache_control": {"type": "ephemeral"},
                },
                {"type": "text", "text": "Question: q?"},
            ],
        }
    ]


def test_cache_tokens_are_costed() -> None:
    message = make_message()
    message.usage.cache_creation_input_tokens = 10_000
    message.usage.cache_read_input_tokens = 20_000

    completion = make_model(FakeAnthropic(message)).complete("s", "p", cached_context="ctx")

    # 1000 in x $4 + 500 out x $20 + 10,000 write x $5 + 20,000 read x $0.20, all / 1M
    assert completion.cost.cost_usd == pytest.approx(0.004 + 0.010 + 0.050 + 0.004)
    assert (completion.cost.cache_write_tokens, completion.cost.cache_read_tokens) == (
        10_000,
        20_000,
    )


def test_token_counter_sends_one_count_request() -> None:
    calls: list[dict[str, Any]] = []

    def count_tokens(**kwargs: Any) -> SimpleNamespace:
        calls.append(kwargs)
        return SimpleNamespace(input_tokens=1234)

    client = SimpleNamespace(messages=SimpleNamespace(count_tokens=count_tokens))

    tokens = ClaudeTokenCounter(cast(Any, client), "claude-opus-5-5").count_tokens(
        "sys", ["<excerpts>x</excerpts>", "Question: q?"]
    )

    assert tokens == 1234
    [call] = calls
    assert call["model"] == "claude-opus-5-5"
    assert call["system"] == "sys"
    assert call["messages"][0]["content"] == [
        {"type": "text", "text": "<excerpts>x</excerpts>"},
        {"type": "text", "text": "Question: q?"},
    ]


def test_missing_credentials_become_a_clear_error() -> None:
    with pytest.raises(MissingCredentialsError, match="ANTHROPIC_API_KEY"), credentials_checked():
        raise TypeError('"Could not resolve authentication method. Expected one of api_key"')


def test_other_type_errors_are_not_hidden() -> None:
    with pytest.raises(TypeError, match="unexpected keyword"), credentials_checked():
        raise TypeError("create() got an unexpected keyword argument 'foo'")
