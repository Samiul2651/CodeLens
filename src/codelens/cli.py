"""Command line.

uv run codelens ingest <path-to-repo> --repo-id my-repo
"""

import argparse
import sys
from pathlib import Path

from qdrant_client import QdrantClient

from codelens.chunking.chunkers import TreeSitterChunker
from codelens.config import Settings, get_settings
from codelens.embedding.ollama import OllamaEmbedder
from codelens.ingestion.service import IngestionService
from codelens.ingestion.state import JsonFileStateStore
from codelens.vectorstore.qdrant import QdrantVectorStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codelens", description="Ask questions about a codebase.")
    commands = parser.add_subparsers(dest="command", required=True)
    ingest = commands.add_parser("ingest", help="index a repo (only changed files are redone)")
    ingest.add_argument("repo", type=Path, help="path to the repo folder")
    ingest.add_argument("--repo-id", help="name to store it under (default: folder name)")
    return parser


def build_ingestion_service(settings: Settings) -> IngestionService:
    return IngestionService(
        chunker=TreeSitterChunker(settings.max_chunk_lines),
        embedder=OllamaEmbedder(settings.ollama_url, settings.embedding_model),
        store=QdrantVectorStore(QdrantClient(url=settings.qdrant_url), settings.collection),
        state=JsonFileStateStore(Path(settings.data_dir)),
        batch_size=settings.embed_batch_size,
        max_file_bytes=settings.max_file_bytes,
    )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if not args.repo.is_dir():
        sys.exit(f"Not a folder: {args.repo}")
    repo_id = args.repo_id or args.repo.resolve().name
    report = build_ingestion_service(get_settings()).ingest(repo_id, args.repo)
    print(
        f"{repo_id}: {report.files_seen} files, {report.files_changed} changed, "
        f"{report.files_deleted} deleted -> {report.chunks_written} chunks "
        f"in {report.embed_requests} embedding requests"
    )


if __name__ == "__main__":
    main()
