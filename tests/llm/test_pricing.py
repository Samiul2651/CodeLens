import pytest

from codelens.llm.pricing import cost_usd


def test_opus_cost() -> None:
    # 1,000 input x $4 / 1M = $0.004  +  500 output x $20 / 1M = $0.010  ->  $0.014
    assert cost_usd("claude-opus-5-5", 1_000, 500) == pytest.approx(0.014)


def test_cache_tokens() -> None:
    # 10,000 cache write x $5 / 1M = $0.05  +  10,000 cache read x $0.20 / 1M = $0.002
    assert cost_usd("claude-opus-5-5", 0, 0, 10_000, 10_000) == pytest.approx(0.052)


def test_unknown_model_is_an_error() -> None:
    with pytest.raises(KeyError):
        cost_usd("claude-unknown", 1, 1)
