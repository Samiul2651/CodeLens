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


def test_ask_mode_option() -> None:
    from codelens.core.models import AnswerMode

    parser = build_parser()

    assert parser.parse_args(["ask", "q", "--repo-id", "d"]).mode is AnswerMode.RAG
    args = parser.parse_args(["ask", "q", "--repo-id", "d", "--mode", "long_context"])
    assert args.mode is AnswerMode.LONG_CONTEXT
    with pytest.raises(SystemExit):
        parser.parse_args(["ask", "q", "--repo-id", "d", "--mode", "magic"])


def test_long_context_answerer_is_wired() -> None:
    from codelens.answering.long_context import LongContextAnswerer
    from codelens.wiring import build_long_context_answerer

    settings = Settings(anthropic_api_key=SecretStr("test-key-not-real"))

    assert isinstance(build_long_context_answerer(settings), LongContextAnswerer)


def test_eval_commands_parse(tmp_path: object) -> None:
    from codelens.core.models import AnswerMode

    parser = build_parser()
    run = parser.parse_args(
        ["eval", "run", "q.json", "--out", "r.csv", "--modes", "rag", "--max-cost", "2.5",
         "--no-judge", "--runs", "3"]
    )  # fmt: skip
    assert (run.eval_command, run.modes, run.max_cost, run.no_judge, run.runs) == (
        "run", [AnswerMode.RAG], 2.5, True, 3,
    )  # fmt: skip
    default = parser.parse_args(["eval", "run", "q.json", "--out", "r.csv"])
    assert default.modes == list(AnswerMode)
    assert parser.parse_args(["eval", "summarize", "r.csv"]).eval_command == "summarize"


def test_judge_is_wired() -> None:
    from codelens.evaluation.judge import LLMJudge
    from codelens.wiring import build_judge

    assert isinstance(
        build_judge(Settings(anthropic_api_key=SecretStr("test-key-not-real"))), LLMJudge
    )
