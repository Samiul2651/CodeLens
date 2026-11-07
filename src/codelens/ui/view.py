"""What the demo page shows, as plain data. No Streamlit here, so it is easy to test."""

from dataclasses import dataclass

from codelens.api.schemas import AskResponse, CostResponse
from codelens.core.models import AnswerMode

MODE_LABELS = {
    AnswerMode.RAG: "RAG: closest chunks only",
    AnswerMode.LONG_CONTEXT: "Long-context: whole repo",
}


@dataclass(frozen=True)
class SourceRow:
    source: str  # path:start-end
    cited: bool  # the answer cites it


def source_rows(response: AskResponse) -> list[SourceRow]:
    """Chunks shown to Claude, best first. Long-context sends every chunk, so it lists
    only the cited ones."""
    cited = set(response.cited)
    if response.mode is AnswerMode.LONG_CONTEXT:
        return [SourceRow(source, cited=True) for source in response.cited]
    return [SourceRow(source, source in cited) for source in response.sources]


def notes(response: AskResponse) -> list[str]:
    """Warnings to show above the answer."""
    found: list[str] = []
    if response.refused:
        found.append("Claude declined to answer this question.")
    if response.context_truncated:
        found.append("The repo did not fit the token budget: some files were left out.")
    return found


def cost_rows(cost: CostResponse) -> list[tuple[str, str]]:
    return [
        ("Model", cost.model),
        ("Input tokens", f"{cost.input_tokens:,}"),
        ("Output tokens", f"{cost.output_tokens:,}"),
        ("Cache write tokens", f"{cost.cache_write_tokens:,}"),
        ("Cache read tokens", f"{cost.cache_read_tokens:,}"),
        ("Cost", f"${cost.cost_usd:.4f}"),
        ("Time", f"{cost.latency_seconds:.1f} s"),
    ]
