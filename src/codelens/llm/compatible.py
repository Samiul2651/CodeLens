"""Other providers' Anthropic-format APIs (DeepSeek) as a LanguageModel and TokenCounter.

Same request shape as Claude, minus the Claude-only parts:
- no beta header, no server-side fallback, no effort setting;
- no prompt caching (DeepSeek ignores cache_control, so the context is sent as plain text);
- no token-count endpoint, so tokens are estimated from text length.
"""

import math
import time
from collections.abc import Callable
from typing import Any

import anthropic

from codelens.core.models import Completion
from codelens.llm.claude import to_completion
from codelens.llm.pricing import price_key


class CompatibleLanguageModel:
    def __init__(
        self,
        client: anthropic.Anthropic,
        model: str,
        max_tokens: int,
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        if price_key(model) is None:
            raise ValueError(f"no price known for {model!r}: add it to llm/pricing.py")
        self._client = client
        self._model = model
        self._max_tokens = max_tokens
        self._clock = clock

    def complete(self, system: str, prompt: str, cached_context: str | None = None) -> Completion:
        content: Any = (
            prompt
            if cached_context is None
            else [{"type": "text", "text": cached_context}, {"type": "text", "text": prompt}]
        )
        started = self._clock()
        message = self._client.messages.create(
            model=self._model,
            max_tokens=self._max_tokens,
            system=system,
            messages=[{"role": "user", "content": content}],
        )
        return to_completion(message, self._clock() - started)


class EstimatedTokenCounter:
    """Input tokens estimated from character count, for APIs without a token-count endpoint.

    tokens = ceil(characters / CHARS_PER_TOKEN x SAFETY_MARGIN). English prose is about 4
    characters per token and code nearer 3, so 3.5 with a 10% margin over-counts slightly.
    That keeps the long-context packer under its budget, which is the side to err on.
    """

    CHARS_PER_TOKEN = 3.5
    SAFETY_MARGIN = 1.10

    def count_tokens(self, system: str, user_texts: list[str]) -> int:
        characters = len(system) + sum(len(text) for text in user_texts)
        # Round off float noise first (350 / 3.5 x 1.1 is 110.00000000000001), then round up.
        return math.ceil(round(characters / self.CHARS_PER_TOKEN * self.SAFETY_MARGIN, 6))
