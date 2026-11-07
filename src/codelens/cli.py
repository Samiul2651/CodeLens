"""Command line.

uv run codelens ingest <path-to-repo> --repo-id my-repo
uv run codelens ask "where is the password checked?" --repo-id my-repo
"""

import argparse
import sys
from pathlib import Path

from codelens.config import get_settings
from codelens.ingestion.state import validate_repo_id
from codelens.wiring import build_cost_log, build_ingestion_service, build_rag_answerer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codelens", description="Ask questions about a codebase.")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="index a repo (only changed files are redone)")
    ingest.add_argument("repo", type=Path, help="path to the repo folder")
    ingest.add_argument("--repo-id", help="name to store it under (default: folder name)")

    ask = commands.add_parser("ask", help="ask a question about an indexed repo (RAG mode)")
    ask.add_argument("question")
    ask.add_argument("--repo-id", required=True)
    return parser


def _ingest(repo: Path, repo_id: str | None) -> None:
    if not repo.is_dir():
        sys.exit(f"Not a folder: {repo}")
    repo_id = repo_id or repo.resolve().name
    report = build_ingestion_service(get_settings()).ingest(repo_id, repo)
    print(
        f"{repo_id}: {report.files_seen} files, {report.files_changed} changed, "
        f"{report.files_deleted} deleted -> {report.chunks_written} chunks "
        f"in {report.embed_requests} embedding requests"
    )


def _ask(question: str, repo_id: str) -> None:
    settings = get_settings()
    answer = build_rag_answerer(settings).answer(validate_repo_id(repo_id), question)
    print(answer.text)
    if answer.sources:
        print("\nSources shown to the model (* = cited):")
        for source in answer.sources:
            print(f"  {'*' if source in answer.cited else ' '} {source}")
    if answer.cost:
        c = answer.cost
        print(
            f"\n[{c.model}  in={c.input_tokens} out={c.output_tokens}  "
            f"${c.cost_usd:.4f}  {c.latency_seconds:.1f}s  "
            f"logged total ${build_cost_log(settings).total_cost():.4f}]"
        )


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        _ingest(args.repo, args.repo_id)
    else:
        _ask(args.question, args.repo_id)


if __name__ == "__main__":
    main()
