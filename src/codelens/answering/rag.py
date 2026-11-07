"""RAG mode: embed the question, find the closest chunks, answer from those only."""

from codelens.answering.prompts import SYSTEM_PROMPT, build_prompt, cited_sources
from codelens.core.interfaces import CostLog, Embedder, LanguageModel, VectorStore
from codelens.core.models import Answer, AnswerMode, CostLogEntry

NOTHING_INDEXED = "Nothing is indexed for this repo yet. Run `codelens ingest` first."


class RagAnswerer:
    def __init__(
        self,
        embedder: Embedder,
        store: VectorStore,
        model: LanguageModel,
        cost_log: CostLog,
        top_k: int,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be >= 1")
        self._embedder = embedder
        self._store = store
        self._model = model
        self._cost_log = cost_log
        self._top_k = top_k

    def answer(self, repo_id: str, question: str) -> Answer:
        [vector] = self._embedder.embed([question])  # one request
        retrieved = self._store.search(repo_id, vector, limit=self._top_k)
        if not retrieved:
            return Answer(text=NOTHING_INDEXED, mode=AnswerMode.RAG)  # no LLM call, no cost

        chunks = [r.chunk for r in retrieved]
        sources = [c.source for c in chunks]
        completion = self._model.complete(SYSTEM_PROMPT, build_prompt(chunks, question))
        self._cost_log.record(CostLogEntry("rag", repo_id, question, completion.cost))

        if completion.refused:
            text = f"Claude declined to answer (category: {completion.refusal_category})."
            return Answer(
                text, AnswerMode.RAG, sources, cost=completion.cost, refused=True
            )  # fmt: skip
        return Answer(
            text=completion.text,
            mode=AnswerMode.RAG,
            sources=sources,
            cited=cited_sources(completion.text, sources),
            cost=completion.cost,
        )
