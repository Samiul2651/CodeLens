from types import SimpleNamespace
from typing import Any, cast

import pytest
from pydantic import SecretStr

from codelens.config import Settings
from codelens.llm.claude import MissingCredentialsError
from codelens.llm.compatible import CompatibleLanguageModel, EstimatedTokenCounter
from codelens.llm.pricing import cost_usd, price_key
from codelens.wiring import (
    _language_model,
    _token_counter,
    build_judge,
    build_long_context_answerer,
    build_rag_answerer,
)


def make_message(model: str = "deepseek-flash") -> SimpleNamespace:
    return SimpleNamespace(
        content=[
            SimpleNamespace(type="thinking", thinking="..."),
            SimpleNamespace(type="text", text="The handler is in app.py."),
        ],
        model=model,
        stop_reason="end_turn",
        stop_details=None,
        usage=SimpleNamespace(
            input_tokens=10_000,
            output_tokens=1_000,
            cache_creation_input_tokens=None,
            cache_read_input_tokens=None,
        ),
    )


class FakeCompatibleClient:
    """Only the plain messages.create endpoint, like DeepSeek's Anthropic-format API."""

    def __init__(self, message: SimpleNamespace) -> None:
        self.calls: list[dict[str, Any]] = []
        self._message = message
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.calls.append(kwargs)
        return self._message


def make_model(client: FakeCompatibleClient) -> CompatibleLanguageModel:
    times = iter([5.0, 7.0])
    return CompatibleLanguageModel(
        cast(Any, client), "deepseek-flash", 4_000, clock=lambda: next(times)
    )


def test_sends_a_plain_request_without_claude_only_options() -> None:
    client = FakeCompatibleClient(make_message())

    make_model(client).complete("You answer questions about code.", "Where is login?")

    call = client.calls[0]
    assert set(call) == {"model", "max_tokens", "system", "messages"}
    assert call["model"] == "deepseek-flash"
    assert call["messages"] == [{"role": "user", "content": "Where is login?"}]


def test_context_is_sent_as_plain_text_without_cache_control() -> None:
    client = FakeCompatibleClient(make_message())

    make_model(client).complete("system", "question", cached_context="whole repo")

    content = client.calls[0]["messages"][0]["content"]
    assert content == [{"type": "text", "text": "whole repo"}, {"type": "text", "text": "question"}]


def test_returns_text_cost_and_latency() -> None:
    completion = make_model(FakeCompatibleClient(make_message())).complete("s", "q")

    assert completion.text == "The handler is in app.py."
    assert completion.cost.latency_seconds == 2.0
    # 10,000 input x $0.30/M + 1,000 output x $1.20/M = $0.003 + $0.0012
    assert completion.cost.cost_usd == pytest.approx(0.0042)


def test_versioned_model_name_in_the_reply_is_priced_as_its_family() -> None:
    completion = make_model(
        FakeCompatibleClient(make_message(model="deepseek-v4.1-flash"))
    ).complete("s", "q")

    assert completion.cost.model == "deepseek-v4.1-flash"
    assert completion.cost.cost_usd == pytest.approx(0.0042)


def test_unknown_model_is_rejected_up_front() -> None:
    with pytest.raises(ValueError, match="no price known"):
        CompatibleLanguageModel(cast(Any, object()), "some-unknown-model", 100)


def test_price_key_maps_known_names_only() -> None:
    assert price_key("claude-opus-5-5") == "claude-opus-5-5"
    assert price_key("deepseek-flash") == "deepseek-flash"
    assert price_key("deepseek-v4.1-flash") == "deepseek-flash"
    assert price_key("deepseek-v4-pro") is None
    with pytest.raises(KeyError):
        cost_usd("deepseek-v4-pro", 1, 1)


def test_estimated_tokens_overcount_slightly() -> None:
    # 70 + 280 = 350 characters / 3.5 = 100 tokens, x 1.10 margin = 110
    tokens = EstimatedTokenCounter().count_tokens("s" * 70, ["a" * 200, "b" * 80])

    assert tokens == 110


def deepseek_settings() -> Settings:
    return Settings(provider="deepseek", deepseek_api_key=SecretStr("test-key-not-real"))


def test_deepseek_settings_wire_the_compatible_model_and_estimator() -> None:
    settings = deepseek_settings()
    client = cast(Any, object())

    assert isinstance(_language_model(settings, client), CompatibleLanguageModel)
    assert isinstance(_language_model(settings, client, judge=True), CompatibleLanguageModel)
    assert isinstance(_token_counter(settings, client), EstimatedTokenCounter)


def test_deepseek_client_points_at_deepseek() -> None:
    from codelens.wiring import _anthropic_client

    client = _anthropic_client(deepseek_settings())

    assert str(client.base_url).startswith("https://api.deepseek.com/anthropic")


def test_deepseek_answerers_and_judge_build() -> None:
    settings = deepseek_settings()

    build_rag_answerer(settings)
    build_long_context_answerer(settings)
    build_judge(settings)


def test_deepseek_without_a_key_says_which_key_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)

    with pytest.raises(MissingCredentialsError, match="DEEPSEEK_API_KEY"):
        build_rag_answerer(Settings(provider="deepseek", _env_file=None))
