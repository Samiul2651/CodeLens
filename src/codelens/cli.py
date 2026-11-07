"""Command line.

uv run codelens ingest <path-to-repo> --repo-id my-repo
uv run codelens ask "where is the password checked?" --repo-id my-repo
uv run codelens ask "how is the app structured?" --repo-id my-repo --mode long_context
"""

import argparse
import sys
from pathlib import Path

from codelens.config import get_settings
from codelens.core.models import Answer, AnswerMode
from codelens.ingestion.state import validate_repo_id
from codelens.wiring import build_answerer, build_cost_log, build_ingestion_service

MAX_SOURCES_PRINTED = 20


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="codelens", description="Ask questions about a codebase.")
    commands = parser.add_subparsers(dest="command", required=True)

    ingest = commands.add_parser("ingest", help="index a repo (only changed files are redone)")
    ingest.add_argument("repo", type=Path, help="path to the repo folder")
    ingest.add_argument("--repo-id", help="name to store it under (default: folder name)")

    ask = commands.add_parser("ask", help="ask a question about an indexed repo")
    ask.add_argument("question")
    ask.add_argument("--repo-id", required=True)
    ask.add_argument(
        "--mode",
        type=AnswerMode,
        choices=list(AnswerMode),
        default=AnswerMode.RAG,
        help="rag (closest chunks) or long_context (everything that fits)",
    )
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


def _print_answer(answer: Answer, total_logged: float) -> None:
    print(answer.text)
    if answer.context_truncated:
        print("\n(Some files did not fit the context budget.)")
    if answer.sources:
        print(f"\n{len(answer.sources)} chunks shown to the model; cited:")
        for source in answer.cited[:MAX_SOURCES_PRINTED]:
            print(f"  * {source}")
    if answer.cost:
        c = answer.cost
        print(
            f"\n[{answer.mode}  {c.model}  in={c.input_tokens} "
            f"cache_write={c.cache_write_tokens} cache_read={c.cache_read_tokens} "
            f"out={c.output_tokens}  ${c.cost_usd:.4f}  {c.latency_seconds:.1f}s  "
            f"logged total ${total_logged:.4f}]"
        )


def _ask(question: str, repo_id: str, mode: AnswerMode) -> None:
    settings = get_settings()
    answer = build_answerer(settings, mode).answer(validate_repo_id(repo_id), question)
    _print_answer(answer, build_cost_log(settings).total_cost())


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "ingest":
        _ingest(args.repo, args.repo_id)
    else:
        _ask(args.question, args.repo_id, args.mode)


if __name__ == "__main__":
    main()
