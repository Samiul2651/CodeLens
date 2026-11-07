# One image for the API and the demo page (compose picks the command).
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.7 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy PATH="/app/.venv/bin:$PATH"

# Dependencies first: this layer is reused until uv.lock changes.
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-default-groups --group ui --no-install-project

COPY src ./src
COPY .streamlit ./.streamlit
RUN uv sync --locked --no-default-groups --group ui

# Run as a normal user, not root.
RUN useradd --create-home codelens && mkdir -p /app/.codelens && chown codelens /app/.codelens
USER codelens

EXPOSE 8000 8501
CMD ["uvicorn", "codelens.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
