"""Talks to the CodeLens HTTP API. The demo page uses only this, never the answerers
directly, so it shows exactly what any other API client would get."""

import httpx
from pydantic import ValidationError

from codelens.api.schemas import AskRequest, AskResponse, ReposResponse
from codelens.core.models import AnswerMode

LIST_TIMEOUT_SECONDS = 10.0
ASK_TIMEOUT_SECONDS = 300.0  # a long-context answer on a big repo can take minutes

# Used when the API gives no plain text reason. The API keeps internal details in its log.
STATUS_MESSAGES = {
    422: "The API did not accept the question or the repo id.",
    502: "The Claude call failed. See the API log.",
    503: "A backend service is not reachable. Start it with `docker compose up -d`.",
}


class ApiError(Exception):
    """A problem with a message that is safe to show on the page."""


class CodeLensClient:
    def __init__(self, http: httpx.Client) -> None:
        self._http = http

    def repos(self) -> list[str]:
        body = self._send("GET", "/repos", timeout=LIST_TIMEOUT_SECONDS)
        return ReposResponse.model_validate(body).repos

    def ask(self, repo_id: str, question: str, mode: AnswerMode) -> AskResponse:
        try:
            request = AskRequest(repo_id=repo_id, question=question, mode=mode)
        except ValidationError as error:  # checked here first: no call for a bad question
            raise ApiError(STATUS_MESSAGES[422]) from error
        body = self._send(
            "POST", "/ask", json=request.model_dump(mode="json"), timeout=ASK_TIMEOUT_SECONDS
        )
        return AskResponse.model_validate(body)

    def _send(self, method: str, path: str, *, timeout: float, json: object = None) -> object:
        try:
            response = self._http.request(method, path, json=json, timeout=timeout)
        except httpx.TimeoutException as error:
            raise ApiError(f"The API did not answer within {timeout:.0f} seconds.") from error
        except httpx.TransportError as error:
            raise ApiError(
                f"Cannot reach the CodeLens API at {self._http.base_url}. "
                "Start it with `uv run uvicorn codelens.api.app:app`."
            ) from error
        if response.is_error:
            raise ApiError(_error_message(response))
        return response.json()


def _error_message(response: httpx.Response) -> str:
    """The API's own reason when it is plain text (FastAPI `detail`), else a fixed message."""
    try:
        body = response.json()
    except ValueError:  # not JSON
        body = None
    detail = body.get("detail") if isinstance(body, dict) else None
    if isinstance(detail, str):
        return f"{detail}."
    code = response.status_code
    return STATUS_MESSAGES.get(code, f"The API returned HTTP {code}.")


def connect(base_url: str) -> CodeLensClient:
    return CodeLensClient(httpx.Client(base_url=base_url))
