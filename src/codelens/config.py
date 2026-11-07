"""Settings from environment variables (prefix CODELENS_) or a .env file.

Secrets like ANTHROPIC_API_KEY are read by the Anthropic SDK itself and never stored here.
"""

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CODELENS_", env_file=".env", extra="ignore")

    app_name: str = "CodeLens"
    qdrant_url: str = "http://localhost:6333"
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    claude_model: str = "claude-opus-5-5"

    # Ingestion
    collection: str = "codelens_chunks"
    data_dir: str = ".codelens"  # local state, e.g. file hashes per repo
    max_chunk_lines: int = 40  # longer functions are split
    embed_batch_size: int = 64  # chunks per embedding request
    max_file_bytes: int = 500_000  # bigger files are skipped (generated code, data)


@lru_cache
def get_settings() -> Settings:
    return Settings()
