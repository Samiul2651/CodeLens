from fastapi.testclient import TestClient

from codelens import __version__
from codelens.api.app import create_app
from codelens.config import Settings


def test_health() -> None:
    with TestClient(create_app(Settings(app_name="Test"))) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": __version__}


def test_app_title_comes_from_settings() -> None:
    assert create_app(Settings(app_name="Custom")).title == "Custom"
