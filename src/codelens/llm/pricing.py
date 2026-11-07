"""US dollars per 1M tokens.

Source: Anthropic first-party API prices as listed in the claude-api skill (cached 2026-10-06).
Prices change: check https://www.anthropic.com/pricing before trusting an estimate.
Cache write = 1.25 x input price (5-minute cache).
"""

from dataclasses import dataclass

TOKENS_PER_PRICE_UNIT = 1_000_000


@dataclass(frozen=True)
class ModelPrices:
    input_per_mtok: float
    output_per_mtok: float
    cache_write_per_mtok: float
    cache_read_per_mtok: float


PRICES: dict[str, ModelPrices] = {
    "claude-opus-5-5": ModelPrices(4.00, 20.00, 5.00, 0.20),
    "claude-sonnet-5-5": ModelPrices(2.00, 10.00, 2.50, 0.20),
    "claude-haiku-5-5": ModelPrices(0.10, 0.50, 0.125, 0.01),
}


def cost_usd(
    model: str, input_tokens: int, output_tokens: int, cache_write: int = 0, cache_read: int = 0
) -> float:
    """Dollar cost of one call. Raises KeyError for a model without a known price,
    so a missing price is noticed instead of silently logged as $0."""
    prices = PRICES[model]
    return (
        input_tokens * prices.input_per_mtok
        + output_tokens * prices.output_per_mtok
        + cache_write * prices.cache_write_per_mtok
        + cache_read * prices.cache_read_per_mtok
    ) / TOKENS_PER_PRICE_UNIT
