import pytest

from codelens.cli import build_ingestion_service, build_parser
from codelens.config import Settings
from codelens.ingestion.service import IngestionService


def test_ingest_command_parses() -> None:
    args = build_parser().parse_args(["ingest", "some/repo", "--repo-id", "demo"])

    assert (args.command, args.repo.as_posix(), args.repo_id) == ("ingest", "some/repo", "demo")


def test_a_command_is_required() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args([])


def test_service_is_wired_from_settings() -> None:
    assert isinstance(build_ingestion_service(Settings()), IngestionService)
