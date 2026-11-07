# CodeLens — Full Plan

Last updated: 2026-11-07 (switched to Python)

## 1. What we are building

CodeLens answers questions about a code repository.
You give it a repo. You ask "Where is the login checked?" It answers and points to the files.

It has two jobs:

1. **Portfolio project**. It must be clean, tested, and easy to demo.
2. **Test setup for the paper**: *RAG vs. long-context LLMs for understanding code repositories*.
   CodeLens must be able to answer the same question in two ways and measure both.

## 2. The two answer modes

| Mode | How it works | Good at | Weak at |
|---|---|---|---|
| **RAG** | Split code into chunks → turn chunks into vectors → store in Qdrant → for each question, find the closest chunks → send only those to Claude | Big repos, low cost per question | Misses code that is spread across many files |
| **Long-context (LC)** | Put as much of the repo as fits straight into Claude's prompt | Questions that need the "big picture" | Cost, speed, repos too big to fit |

The paper compares these two. So both modes must use **the same questions, same model, same scoring**.

## 3. How the system fits together

```
            ┌─────────────── INGEST (once per repo) ───────────────┐
 Git repo → │ File walker → Chunker → Embedder (Ollama) → Qdrant   │
            └──────────────────────────────────────────────────────┘

            ┌─────────────── ASK (per question) ───────────────────┐
 Question → │ RAG mode: embed question → search Qdrant → top chunks│
            │ LC mode:  pack repo files into prompt                │ → Claude → Answer + file refs
            └──────────────────────────────────────────────────────┘

            ┌─────────────── EVALUATE (for the paper) ─────────────┐
 Question   │ Run both modes → score answers → log tokens, cost,   │
 set      → │ time → save results to CSV                           │
            └──────────────────────────────────────────────────────┘
```

## 4. Tech stack

| Part | Choice | Why |
|---|---|---|
| Language | Python 3.12 | AI/ML job market; best RAG and research tools |
| API | FastAPI + Pydantic | Standard for AI backends; typed request/response models |
| Vector DB | Qdrant (Docker) + `qdrant-client` | Fast, free, good Python client |
| Embeddings | Ollama, small model, local | $0 API cost; fits in 4 GB VRAM |
| LLM | Claude via `anthropic` SDK | Cloud credits; long context window for LC mode |
| Code parsing | tree-sitter (`tree-sitter-language-pack`) | One tool for many languages; gives functions and classes |
| Token counting | Claude token-count endpoint | Exact counts for the cost log and LC packing |
| Analysis (paper) | pandas, scipy, matplotlib | Stats and charts for the paper |
| Demo UI (late) | Streamlit | A demo page in ~100 lines of Python |
| Tooling | `uv`, `pytest`, `ruff`, `mypy` | Fast installs, tests, lint, type checks |
| Run everything | Docker Compose | One command to start API + Qdrant + Ollama |

We do **not** use LangChain or LlamaIndex. Writing each step yourself means you understand it, can measure it fairly for the paper, and can explain it in interviews.

## 5. Code layout

Each module has one job. `core` has no outside libraries: only data models and interfaces (Python `Protocol` classes).

```
codelens/
├─ pyproject.toml
├─ docker-compose.yml
├─ src/codelens/
│  ├─ core/          # Models (Chunk, Answer, CostRecord) + Protocols: Chunker, Embedder, VectorStore, AnswerGenerator, ContextPacker
│  ├─ chunking/      # LineWindowChunker, TreeSitterChunker
│  ├─ embedding/     # OllamaEmbedder (batched)
│  ├─ vectorstore/   # QdrantVectorStore
│  ├─ llm/           # ClaudeAnswerGenerator, cost tracking
│  ├─ ingestion/     # Walk repo → chunk → embed in batches → save
│  ├─ answering/     # RagAnswerer, LongContextAnswerer
│  ├─ api/           # FastAPI app: GET /repos, POST /ask
│  ├─ evaluation/    # Run question sets, score, write CSV
│  └─ ui/            # Streamlit demo page (a client of the API)
├─ analysis/         # Notebooks/scripts for paper tables and charts
└─ tests/            # Unit tests + Qdrant tests (Testcontainers)
```

Rules we follow:
- Depend on Protocols, not concrete classes (SOLID). Swap Qdrant or the embedder without touching the rest.
- Embed in **batches**, not one call per chunk. Upsert to Qdrant in batches too.
- No API keys in code or Git. Use a `.env` file that is in `.gitignore`.
- Never send private repo code anywhere except the Claude API.
- Type hints everywhere; `mypy` and `ruff` run in CI.

## 6. Learning path (practice projects first)

You want to understand each part before building it. Each practice project is small, its own repo, and teaches one thing.

| # | Practice project | You learn | Status |
|---|---|---|---|
| 1 | `embedding-playground` (Python + Ollama) | What embeddings are, cosine similarity | ✅ Done |
| 2 | `qdrant-playground` | Run Qdrant in Docker, create a collection, insert vectors in batches, search, filter by payload | ✅ Done (task 7 needs Ollama to test) |
| 3 | `tree-sitter-code-chunker` | Parse Python/JS/Java/C# files, cut them into function/class chunks with file path + line numbers | ✅ Done |
| 4 | `claude-api-basics` | Call Claude, count tokens, use prompt caching, log cost per call | ✅ Done |
| 5 | `fastapi-basics` | Build a small typed API with Pydantic models, dependency injection, and tests | ✅ Done |
| 6 | `mini-rag-console` | Glue 1–4 together: index one folder, ask one question in the terminal | ✅ Done (not yet run end to end: needs Ollama + API key) |

Projects 3–6 are written as lesson + solution: learn them by deleting a solution and
rewriting it against its tests.

After #6 you have seen every piece. Then we build CodeLens itself.

## 7. Build phases for CodeLens

Each phase ends with something that runs and has tests.

**Phase 0 — Setup (1 day)** ✅ Done 2026-11-07
- `uv init`, folder layout from section 5, `ruff` + `mypy` + `pytest` config.
- Docker Compose with Qdrant + Ollama.
- GitHub repo, README stub, GitHub Actions CI (lint + type check + test on push).

**Phase 1 — Ingestion (1 week)** ✅ Done 2026-11-07 (real run: fastapi-basics, 21 files -> 71 chunks in 2 Ollama requests, ~20 s; re-run with no changes ~5 s)
- File walker: skip `.git/`, `node_modules/`, `venv/`, build folders, binaries, files in `.gitignore`.
- Chunkers: `LineWindowChunker` (any language) and `TreeSitterChunker` (functions/classes).
- Each chunk keeps: repo id, file path, start/end line, language, symbol name.
- Batch embed → batch upsert to Qdrant.
- Re-index only changed files (hash per file).

**Phase 2 — RAG answers (1 week)** ✅ Done 2026-11-07 (Claude call tested with a fake client; real call needs an API key)
- `POST /ask` with `mode = "rag"`.
- Embed question → top-k search → build prompt → Claude → answer with file + line refs.
- Log for every call: input tokens, output tokens, time, cost.

**Phase 3 — Long-context answers (3–4 days)** ✅ Done 2026-11-07 (uses the same stored chunks as RAG; real Claude call needs an API key)
- `ContextPacker`: put files into the prompt until a token limit is reached.
- Order files by a simple rule (e.g. most important folders first) and record when the repo did not fit.
- Use prompt caching so the same repo context is not paid for in full on every question.
- `POST /ask` with `mode = "long_context"`.

**Phase 4 — Evaluation harness (1–2 weeks, overlaps the paper)** ✅ Harness done 2026-11-07. Dry run (real retrieval, fake Claude) on the 9-question pilot: RAG retrieval recall 1.00 locate / 0.50 explain_flow / 0.83 architecture; long-context 1.00. Real pilot needs an API key (~$0.45).
- Question set format (JSON): question, repo, expected answer, expected files, question type.
- Run each question in both modes. Save results to CSV.
- Scores: did it find the right files (file recall), is the answer correct (LLM judge + your manual check on a sample), tokens, cost, time.
- `analysis/` scripts turn the CSV into paper tables and charts.

**Phase 5 — Portfolio polish (1 week)** ✅ Done 2026-11-08, except the GIF (needs a real answer, so an API key) and the CI badge (added once CI runs on GitHub)
- Streamlit page: pick repo, ask, switch mode, see answer with file refs and cost.
- README with diagram, demo GIF, how to run in one command.
- Clean tests, CI badge.

## 8. Research plan (the paper)

**Working title:** RAG vs. Long-Context LLMs for Understanding Code Repositories: An Empirical Study

**Research questions (draft):**
- RQ1: Which mode answers more correctly? Does it depend on the question type (find one function / explain a flow across files / architecture question)?
- RQ2: What does each mode cost in tokens, money, and time?
- RQ3: How does repo size change the result?
- RQ4 (RAG only): How much do chunking choices matter (line windows vs. tree-sitter chunks, chunk size, top-k)?

**Data:**
- Check existing code-QA benchmarks before making our own (to verify: RepoQA, Long Code Arena, CodeRAG-Bench, and similar). Reusing one saves time and is easier to defend.
- Plus a small own set: about 8–10 open-source repos in several languages and sizes, about 20 questions each, with answers you check by hand.

**Steps:**
1. Literature review: find what is already done, find the gap.
2. Pick benchmark(s) and repos.
3. Small pilot (1 repo, 10 questions) to test the harness and the cost.
4. Full run.
5. Analysis, write-up, pick conference.

## 9. Budget (paper API spend, limit ≈ $50)

We will not guess prices. We read them from Anthropic's pricing page when we get to Practice project 4.

How to estimate cost, per mode:

```
cost per question = (input tokens × input price) + (output tokens × output price)
total cost        = cost per question × number of questions × number of runs
```

- **Input tokens (RAG)** = top-k chunks × tokens per chunk + question. Small.
- **Input tokens (LC)** = size of packed repo in tokens. Large. This is where the money goes.
- **Prompt caching** cuts LC cost a lot: the repo is the same for all ~20 questions on that repo, so we pay full price once and the cheaper cached price after that.
- **Embeddings** are local (Ollama) → $0.
- **LLM judge** calls also cost money. Count them in the budget.

Before the full run, the pilot (step 3 above) gives us real numbers to put in this formula.

## 10. Hardware limits

- RTX 3050, 4 GB VRAM: fine for small embedding models in Ollama. Too small for a good local LLM, so all answers come from Claude.
- If we want to test a bigger embedding model, rent a cloud GPU for a few hours.

## 11. Risks and how we handle them

| Risk | What we do |
|---|---|
| API spend goes over $50 | Pilot first; prompt caching; cap questions per repo; log cost on every call |
| Big repos do not fit in LC mode | Record it as a result (it is a real finding), and pick repo sizes on purpose |
| LLM judge is unreliable | Hand-check a random sample and report how often you agree with the judge |
| Scope grows too big | Portfolio needs Phases 0–3 + 5. Paper needs Phases 0–4. Nothing else |
| Someone already did this exact study | Do the literature review early (step 1) and adjust the angle |
| Python code gets messy | Type hints + `mypy` + `ruff` in CI from day one; strict review against SOLID |

## 12. Next step

Run the pilot (needs an API key, about $0.45), then start the literature review.
