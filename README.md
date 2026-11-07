# CodeLens

Ask questions about a codebase. CodeLens answers in two ways and measures both:

- **RAG** — find the code chunks closest to the question, send only those to Claude.
- **Long-context** — put as much of the repo as fits straight into Claude's prompt.

It is also the test setup for a research paper: *RAG vs. long-context LLMs for
understanding code repositories*. Full plan: [PLAN.md](PLAN.md).

## Status

- **Phase 0 — setup:** done. Layout, core models and interfaces, `/health` API, Docker services, CI.
- **Phase 1 — ingestion:** done. `codelens ingest` walks a repo (respecting `.gitignore`, skipping
  binary and huge files), cuts code into function chunks with tree-sitter, embeds them in batches
  with Ollama and stores them in Qdrant. Re-running only redoes changed files; deleted files lose
  their chunks.
- **Next:** Phase 2 — RAG answers with Claude.

## Stack

Python 3.12 · FastAPI · Qdrant · Ollama (local embeddings) · tree-sitter · Claude ·
uv · pytest · ruff · mypy

## Layout

```
src/codelens/
  core/          models + Protocols (standard library only — a test enforces it)
  chunking/      source files -> chunks                       (Phase 1)
  embedding/     chunks -> vectors (Ollama)                   (Phase 1)
  vectorstore/   Qdrant                                       (Phase 1)
  ingestion/     walk repo, embed in batches, store           (Phase 1)
  llm/           Claude calls, token counting, cost           (Phase 2)
  answering/     RAG and long-context answerers               (Phases 2-3)
  api/           FastAPI app
  evaluation/    question sets, scoring, CSV results          (Phase 4)
analysis/        paper tables and charts
ui/              demo page
tests/
```

## Run

```bash
uv sync
docker compose up -d                       # Qdrant + Ollama
docker compose exec ollama ollama pull nomic-embed-text
copy .env.example .env                     # then add your ANTHROPIC_API_KEY
uv run uvicorn codelens.api.app:app --reload
```

Index a repo (run it again after edits: only changed files are redone):

```bash
uv run codelens ingest path/to/repo --repo-id my-repo
```

## How ingestion works

```
repo -> walk (skip .gitignore, binary, > 500 KB) -> SHA-256 per file
     -> compare with last run's hashes (.codelens/<repo>.files.json)
     -> delete chunks of changed + deleted files (one Qdrant call)
     -> chunk changed files: one chunk per function (tree-sitter), line windows for the rest
     -> embed in batches of 64 (one Ollama request per batch)
     -> upsert each batch (one Qdrant call per batch)
     -> save the new hashes last, so a failed run is simply redone next time
```

API docs: http://127.0.0.1:8000/docs · Qdrant dashboard: http://localhost:6333/dashboard

## Checks (same as CI)

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src tests
uv run pytest
```
