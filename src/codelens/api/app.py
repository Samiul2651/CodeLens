"""FastAPI app factory. Run: `uv run uvicorn codelens.api.app:app --reload`."""

from fastapi import FastAPI
from pydantic import BaseModel

from codelens import __version__
from codelens.config import Settings, get_settings


class HealthResponse(BaseModel):
    status: str
    version: str


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title=settings.app_name, version=__version__)

    @app.get("/health", tags=["health"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    return app


app = create_app()
