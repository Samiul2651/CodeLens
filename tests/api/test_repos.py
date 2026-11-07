from collections.abc import Iterator

import httpx
from fastapi.testclient import TestClient

from codelens.api.app import create_app, get_repo_catalog
from codelens.config import Settings


class StubCatalog:
    def __init__(self, repos: list[str], error: Exception | None = None) -> None:
        self._repos = repos
        self._error = error

    def repo_ids(self) -> list[str]:
        if self._error:
            raise self._error
        return self._repos


def client_with(catalog: StubCatalog) -> Iterator[TestClient]:
    app = create_app(Settings(app_name="Test"))
    app.dependency_overrides[get_repo_catalog] = lambda: catalog
    with TestClient(app) as client:
        yield client


def test_repos_lists_ingested_repo_ids() -> None:
    for client in client_with(StubCatalog(["fastapi-basics", "demo"])):
        response = client.get("/repos")

    assert response.status_code == 200
    assert response.json() == {"repos": ["fastapi-basics", "demo"]}


def test_repos_backend_down_gives_503_without_internal_details() -> None:
    catalog = StubCatalog([], error=httpx.ConnectError("refused by 10.0.0.5"))

    for client in client_with(catalog):
        response = client.get("/repos")

    assert response.status_code == 503
    assert "10.0.0.5" not in response.text
