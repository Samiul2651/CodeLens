"""Build the real parts from settings. The only place that knows every concrete class;
the API and the CLI both use it."""

from pathlib import Path

import anthropic
from qdrant_client import QdrantClient

from codelens.answering.context_packer import ContextPacker
from codelens.answering.long_context import LongContextAnswerer
from codelens.answering.rag import RagAnswerer
from codelens.chunking.chunkers import TreeSitterChunker
from codelens.config import Settings
from codelens.core.interfaces import Answerer
from codelens.core.models import AnswerMode
from codelens.embedding.ollama import OllamaEmbedder
from codelens.evaluation.judge import LLMJudge
from codelens.ingestion.service import IngestionService
from codelens.ingestion.state import JsonFileStateStore
from codelens.llm.claude import ClaudeLanguageModel, ClaudeTokenCounter
from codelens.llm.cost_log import CsvCostLog
from codelens.vectorstore.qdrant import QdrantVectorStore

COST_LOG_FILE = "cost_log.csv"


def _embedder(settings: Settings) -> OllamaEmbedder:
    return OllamaEmbedder(settings.ollama_url, settings.embedding_model)


def _store(settings: Settings) -> QdrantVectorStore:
    return QdrantVectorStore(QdrantClient(url=settings.qdrant_url), settings.collection)


def _anthropic_client(settings: Settings) -> anthropic.Anthropic:
    key = settings.anthropic_api_key
    return anthropic.Anthropic(api_key=key.get_secret_value()) if key else anthropic.Anthropic()


def _model(settings: Settings, client: anthropic.Anthropic) -> ClaudeLanguageModel:
    return ClaudeLanguageModel(
        client, settings.claude_model, settings.claude_effort, settings.claude_max_tokens
    )


def build_ingestion_service(settings: Settings) -> IngestionService:
    return IngestionService(
        chunker=TreeSitterChunker(settings.max_chunk_lines),
        embedder=_embedder(settings),
        store=_store(settings),
        state=JsonFileStateStore(Path(settings.data_dir)),
        batch_size=settings.embed_batch_size,
        max_file_bytes=settings.max_file_bytes,
    )


def build_cost_log(settings: Settings) -> CsvCostLog:
    return CsvCostLog(Path(settings.data_dir) / COST_LOG_FILE)


def build_rag_answerer(settings: Settings) -> RagAnswerer:
    model = _model(settings, _anthropic_client(settings))
    return RagAnswerer(
        _embedder(settings), _store(settings), model, build_cost_log(settings), settings.top_k
    )


def build_long_context_answerer(settings: Settings) -> LongContextAnswerer:
    client = _anthropic_client(settings)
    packer = ContextPacker(
        ClaudeTokenCounter(client, settings.claude_model), settings.context_token_budget
    )
    return LongContextAnswerer(
        _store(settings), _model(settings, client), packer, build_cost_log(settings)
    )


def build_judge(settings: Settings) -> LLMJudge:
    model = ClaudeLanguageModel(
        _anthropic_client(settings), settings.judge_model, settings.judge_effort, 2_000
    )
    return LLMJudge(model)


def build_answerer(settings: Settings, mode: AnswerMode) -> Answerer:
    if mode is AnswerMode.RAG:
        return build_rag_answerer(settings)
    return build_long_context_answerer(settings)
