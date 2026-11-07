"""Prompts shared by the answer modes. One system prompt for both modes, so the paper
compares retrieval, not wording."""

import re

from codelens.core.models import Chunk

SYSTEM_PROMPT = (
    "You answer questions about a code repository using only the code excerpts provided.\n"
    "Each excerpt is in an <excerpt> tag whose source attribute is path:start-end.\n"
    "Cite every excerpt you rely on in square brackets, exactly as its source is written, "
    "for example [app/auth.py:15-17].\n"
    "If the excerpts do not contain the answer, say so plainly instead of guessing.\n"
    "Be concise."
)

_CITATION = re.compile(r"\[([^\[\]\s]+:\d+-\d+)\]")


def format_excerpts(chunks: list[Chunk]) -> str:
    parts = []
    for chunk in chunks:
        symbol = f' symbol="{chunk.symbol}"' if chunk.symbol else ""
        parts.append(
            f'<excerpt source="{chunk.source}" language="{chunk.language}"{symbol}>\n'
            f"{chunk.text}\n</excerpt>"
        )
    return "\n\n".join(parts)


def question_block(question: str) -> str:
    return f"Question: {question}"


def build_prompt(chunks: list[Chunk], question: str) -> str:
    """Excerpts first, question last: the model reads the code, then the question.
    Long-context mode sends the same two parts as separate blocks so the first is cached."""
    return f"<excerpts>\n{format_excerpts(chunks)}\n</excerpts>\n\n{question_block(question)}"


def cited_sources(answer_text: str, shown: list[str]) -> list[str]:
    """Sources the answer cites, in the order shown, ignoring citations of sources
    that were never shown (made-up citations)."""
    cited = set(_CITATION.findall(answer_text))
    return [source for source in shown if source in cited]
