"""Claude as a LanguageModel: one call in, text + tokens + cost + latency out."""

import time
from collections.abc import Callable
from typing import Any

import anthropic

from codelens.config import Effort
from codelens.core.models import Completion, CostRecord
from codelens.llm.pricing import PRICES, cost_usd

# Refusal fallback: if a safety check declines, the API retries on a suitable model
# inside the same call (beta).
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class ClaudeLanguageModel:
    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        effort: Effort,
        max_tokens: int,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if model not in PRICES:
            raise ValueError(f"no price known for {model!r}: add it to llm/pricing.py")
        self._client = client
        self._model = model
        self._effort: Effort = effort
        self._max_tokens = max_tokens
        self._clock = clock

    def complete(self, system: str, prompt: str) -> Completion:
        started = self._clock()
        message = self._client.beta.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            output_config={"effort": self._effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
            messages=[{"role": "user", "content": prompt}],
        )
        latency = self._clock() - started
        return to_completion(message, latency)


def to_completion(message: Any, latency_seconds: float) -> Completion:
    usage = message.usage
    cache_write = usage.cache_creation_input_tokens or 0
    cache_read = usage.cache_read_input_tokens or 0
    cost = CostRecord(
        model=message.model,  # the model that actually answered (may be a fallback)
        input_tokens=usage.input_tokens,
        output_tokens=usage.output_tokens,
        cache_write_tokens=cache_write,
        cache_read_tokens=cache_read,
        cost_usd=cost_usd(
            message.model, usage.input_tokens, usage.output_tokens, cache_write, cache_read
        ),
        latency_seconds=latency_seconds,
    )
    if message.stop_reason == "refusal":
        details = message.stop_details
        category = details.category if details is not None else None
        return Completion(text="", cost=cost, refused=True, refusal_category=category)
    text = "".join(block.text for block in message.content if block.type == "text")
    return Completion(text=text, cost=cost)
