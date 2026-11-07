"""Settings from environment variables (prefix CODELENS_) or a .env file.

The Anthropic key is read from ANTHROPIC_API_KEY (environment or .env) as a SecretStr,
so it never shows up in logs or reprs. If it is not set, the Anthropic SDK looks for
credentials on its own.
"""

from functools import lru_cache
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

Effort = Literal["low", "medium", "high", "xhigh", "max"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CODELENS_", env_file=".env", extra="ignore")

    app_name: str = "CodeLens"
    qdrant_url: str = "http://localhost:6333"
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"

    # Ingestion
    collection: str = "codelens_chunks"
    data_dir: str = ".codelens"  # local state: file hashes, cost log
    max_chunk_lines: int = 40  # longer functions are split
    embed_batch_size: int = 64  # chunks per embedding request
    max_file_bytes: int = 500_000  # bigger files are skipped (generated code, data)

    # Answers
    claude_model: str = "claude-opus-5-5"
    claude_effort: Effort = "medium"  # Opus 5.5 default; a variable in the paper
    claude_max_tokens: int = 16_000
    top_k: int = 8  # chunks sent to Claude in RAG mode
    # Long-context mode: max input tokens per request. Opus 5.5 takes up to 1M, but cost
    # grows with it: 100,000 tokens x $4 / 1M = $0.40 uncached, x $0.20 / 1M = $0.02 cached.
    context_token_budget: int = 100_000
    anthropic_api_key: SecretStr | None = Field(
        default=None, validation_alias=AliasChoices("ANTHROPIC_API_KEY")
    )


@lru_cache
def get_settings() -> Settings:
    return Settings()
