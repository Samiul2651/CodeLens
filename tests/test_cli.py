import pytest
from pydantic import SecretStr

from codelens.answering.rag import RagAnswerer
from codelens.cli import build_parser
from codelens.config import Settings
from codelens.ingestion.service import IngestionService
from codelens.wiring import build_ingestion_service, build_rag_answerer


def test_ingest_command_parses() -> None:
    args = build_parser().parse_args(["ingest", "some/repo", "--repo-id", "demo"])

    assert (args.command, args.repo.as_posix(), args.repo_id) == ("ingest", "some/repo", "demo")


def test_ask_command_parses() -> None:
    args = build_parser().parse_args(["ask", "where is login?", "--repo-id", "demo"])

    assert (args.command, args.question, args.repo_id) == ("ask", "where is login?", "demo")


def test_ask_needs_a_repo_id() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["ask", "where is login?"])


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_parts_are_wired_from_settings() -> None:
    settings = Settings(anthropic_api_key=SecretStr("test-key-not-real"))

    assert isinstance(build_ingestion_service(settings), IngestionService)
    assert isinstance(build_rag_answerer(settings), RagAnswerer)


def test_api_key_is_hidden_in_reprs() -> None:
    settings = Settings(anthropic_api_key=SecretStr("test-key-not-real"))

    assert "test-key-not-real" not in repr(settings)
