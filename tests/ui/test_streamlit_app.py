"""Runs the real page script with a fake API client (Streamlit's AppTest)."""

from collections.abc import Iterator
from pathlib import Path

import pytest
import streamlit as st
from streamlit.testing.v1 import AppTest

from codelens.api.schemas import AskResponse
from codelens.core.models import AnswerMode
from codelens.ui import client as api

APP = Path(__file__).parents[2] / "src" / "codelens" / "ui" / "streamlit_app.py"


class FakeClient:
    def __init__(self, repos: list[str] | None = None, error: str | None = None) -> None:
        self._repos = repos or []
        self._error = error
        self.asked: list[tuple[str, str, AnswerMode]] = []

    def repos(self) -> list[str]:
        if self._error:
            raise api.ApiError(self._error)
        return self._repos

    def ask(self, repo_id: str, question: str, mode: AnswerMode) -> AskResponse:
        self.asked.append((repo_id, question, mode))
        return AskResponse(
            answer="Ids come from a counter.",
            mode=mode,
            sources=["app/store.py:10-14"],
            cited=["app/store.py:10-14"],
            refused=False,
            context_truncated=False,
            cost=None,
        )


@pytest.fixture
def fresh_cache() -> Iterator[None]:
    st.cache_resource.clear()  # the page caches its client
    yield
    st.cache_resource.clear()


def run_with(monkeypatch: pytest.MonkeyPatch, fake: FakeClient) -> AppTest:
    monkeypatch.setattr(api, "connect", lambda base_url: fake)
    return AppTest.from_file(str(APP)).run()


@pytest.mark.usefixtures("fresh_cache")
def test_api_error_is_shown(monkeypatch: pytest.MonkeyPatch) -> None:
    page = run_with(monkeypatch, FakeClient(error="Cannot reach the CodeLens API"))

    assert page.error[0].value == "Cannot reach the CodeLens API"


@pytest.mark.usefixtures("fresh_cache")
def test_no_repos_shows_the_ingest_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    page = run_with(monkeypatch, FakeClient(repos=[]))

    assert "codelens ingest" in page.info[0].value


@pytest.mark.usefixtures("fresh_cache")
def test_asking_shows_the_answer_and_sources(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = FakeClient(repos=["demo"])
    page = run_with(monkeypatch, fake)

    page.text_area[0].input("  how are ids made?  ")
    page.button[0].click().run()

    assert fake.asked == [("demo", "how are ids made?", AnswerMode.RAG)]
    assert any("Ids come from a counter." in m.value for m in page.markdown)
    assert any("app/store.py:10-14" in m.value for m in page.markdown)
