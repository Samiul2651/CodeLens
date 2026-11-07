"""Long-context mode: send as much of the repo as fits, no retrieval.

It uses the SAME chunks RAG mode searches (read back from the vector store), in file order,
so the paper compares "pick a few chunks" against "show everything that fits" on identical
content and with an identical system prompt.

The repo block is prompt-cached: the first question about a repo pays the cache write
(1.25 x input price), later questions within 5 minutes pay the cache read price.
"""

from codelens.answering.context_packer import ContextPacker
from codelens.answering.prompts import SYSTEM_PROMPT, cited_sources, question_block
from codelens.answering.rag import NOTHING_INDEXED
from codelens.core.interfaces import CostLog, LanguageModel, VectorStore
from codelens.core.models import Answer, AnswerMode, CostLogEntry

NOTHING_FITS = "Not even one file of this repo fits in the context budget."


class LongContextAnswerer:
    def __init__(
        self, store: VectorStore, model: LanguageModel, packer: ContextPacker, cost_log: CostLog
    ) -> None:
        self._store = store
        self._model = model
        self._packer = packer
        self._cost_log = cost_log

    def answer(self, repo_id: str, question: str) -> Answer:
        chunks = self._store.all_chunks(repo_id)
        if not chunks:
            return Answer(text=NOTHING_INDEXED, mode=AnswerMode.LONG_CONTEXT)

        packed = self._packer.pack(chunks, question)
        if not packed.chunks:
            return Answer(text=NOTHING_FITS, mode=AnswerMode.LONG_CONTEXT, context_truncated=True)

        sources = [c.source for c in packed.chunks]
        completion = self._model.complete(
            SYSTEM_PROMPT, question_block(question), cached_context=packed.context
        )
        self._cost_log.record(CostLogEntry("long_context", repo_id, question, completion.cost))

        if completion.refused:
            return Answer(
                text=f"Claude declined to answer (category: {completion.refusal_category}).",
                mode=AnswerMode.LONG_CONTEXT,
                sources=sources,
                cost=completion.cost,
                refused=True,
                context_truncated=packed.truncated,
            )
        return Answer(
            text=completion.text,
            mode=AnswerMode.LONG_CONTEXT,
            sources=sources,
            cited=cited_sources(completion.text, sources),
            cost=completion.cost,
            context_truncated=packed.truncated,
        )
