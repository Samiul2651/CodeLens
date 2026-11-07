"""FastAPI app factory. Run: `uv run uvicorn codelens.api.app:app --reload`."""

import logging
from functools import lru_cache
from typing import Annotated

import anthropic
import httpx
from fastapi import Depends, FastAPI, HTTPException, status
from pydantic import BaseModel
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse

from codelens import __version__
from codelens.api.schemas import AskRequest, AskResponse, ReposResponse
from codelens.config import Settings, get_settings
from codelens.core.interfaces import Answerer, RepoCatalog
from codelens.core.models import AnswerMode
from codelens.llm.claude import MissingCredentialsError
from codelens.wiring import (
    build_long_context_answerer,
    build_rag_answerer,
    build_repo_catalog,
)

logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    status: str
    version: str


# One shared answerer (and its HTTP clients) per mode for the whole app. Tests override these.
@lru_cache
def get_rag_answerer() -> Answerer:
    return build_rag_answerer(get_settings())


@lru_cache
def get_long_context_answerer() -> Answerer:
    return build_long_context_answerer(get_settings())


@lru_cache
def get_repo_catalog() -> RepoCatalog:
    return build_repo_catalog(get_settings())


BACKEND_ERRORS = (httpx.HTTPError, ResponseHandlingException, UnexpectedResponse)


def _backend_unavailable() -> HTTPException:
    # Ollama or Qdrant is down or failed. Log details, return a plain message.
    logger.exception("search backend failed")
    return HTTPException(
        status.HTTP_503_SERVICE_UNAVAILABLE, "search backend unavailable (Ollama or Qdrant)"
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title=settings.app_name, version=__version__)

    @app.get("/health", tags=["health"])
    def health() -> HealthResponse:
        return HealthResponse(status="ok", version=__version__)

    @app.get("/repos", tags=["repos"])
    def repos(catalog: Annotated[RepoCatalog, Depends(get_repo_catalog)]) -> ReposResponse:
        try:
            return ReposResponse(repos=catalog.repo_ids())
        except BACKEND_ERRORS as error:
            raise _backend_unavailable() from error

    @app.post("/ask", tags=["answers"])
    def ask(
        request: AskRequest,
        rag: Annotated[Answerer, Depends(get_rag_answerer)],
        long_context: Annotated[Answerer, Depends(get_long_context_answerer)],
    ) -> AskResponse:
        answerer = rag if request.mode is AnswerMode.RAG else long_context
        try:
            answer = answerer.answer(request.repo_id, request.question)
        except BACKEND_ERRORS as error:
            raise _backend_unavailable() from error
        except MissingCredentialsError as error:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                "Claude credentials are not set: add ANTHROPIC_API_KEY to .env",
            ) from error
        except anthropic.APIError as error:
            logger.exception("Claude call failed")
            raise HTTPException(
                status.HTTP_502_BAD_GATEWAY, "language model call failed"
            ) from error
        return AskResponse.from_answer(answer)

    return app


app = create_app()
