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
- **Phase 2 — RAG answers:** done. `POST /ask` and `codelens ask` embed the question, take the
  8 closest chunks of that repo, and ask Claude to answer from them only, citing `path:start-end`.
  Every call's tokens, cost and latency go to `.codelens/cost_log.csv`.
- **Next:** Phase 3 — long-context answers.

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

Ask a question (needs `ANTHROPIC_API_KEY` in `.env`):

```bash
uv run codelens ask "how are snippet ids assigned?" --repo-id my-repo
```

Or over HTTP:

```bash
curl -X POST http://127.0.0.1:8000/ask -H "Content-Type: application/json" -d "{\"repo_id\": \"my-repo\", \"question\": \"how are snippet ids assigned?\"}"
```

The response has the answer, `sources` (chunks shown to Claude, best first), `cited` (the ones
the answer actually cites — made-up citations are dropped), and `cost` (tokens, dollars, seconds).

## How a RAG answer works

```
question -> Ollama (1 request) -> Qdrant: 8 closest chunks of this repo
         -> Claude: system prompt "answer only from these excerpts, cite path:start-end"
         -> answer + sources + cited + cost  -> one row in .codelens/cost_log.csv
```

Errors: Ollama or Qdrant down -> 503, Claude failure -> 502 (details only in the server log).
A refusal is returned as an answer with `refused: true`, and still logged, because it still costs.

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
