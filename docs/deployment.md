# Local Deployment

This project is set up to run locally with Python in the workspace and supporting services in Docker.

## Prerequisites

- Python 3.12
- `uv`
- Docker with Compose support
- Enough local resources to run Ollama

The checked-in compose file starts:

- `sqlite-init`
- `qdrant`
- `ollama`
- `ollama-pull-models` via the `setup` profile

## Environment

Create a local env file:

```bash
cp .env.example .env
```

Important defaults from the repo:

- `GENERATION__OPENAI_BASE_URL=http://localhost:11434/v1`
- `GENERATION__OPENAI_API_KEY=ollama`
- `GENERATION__GENERATION_MODEL=gemma3:12b`
- `RAG__RAG_MODEL=gemma3:12b`
- `RAG__RAG_ENABLE_LLM_REVIEW=false`
- `RAG__RAG_ENABLE_CROSS_ENCODER_RERANK=true`
- `RAG__RAG_MAX_RETRIEVAL_QUERIES=1`
- `QDRANT__QDRANT_URL=http://localhost:6333`

If you want to use a different OpenAI-compatible endpoint, update `.env` and skip the local Ollama dependency.
Set `RAG__RAG_ENABLE_LLM_REVIEW=true` if you want the groundedness reviewer to call the configured model instead of using the default deterministic citation check.
Set `RAG__RAG_ENABLE_CROSS_ENCODER_RERANK=false` if you want the faster retrieval-score reranker instead of the default local cross-encoder.
Increase `RAG__RAG_MAX_RETRIEVAL_QUERIES` up to `3` when recall is more important than response time.

## Install

```bash
uv sync --dev
```

## Start Services

```bash
docker compose up -d sqlite-init qdrant ollama
docker compose --profile setup up ollama-pull-models
```

## Smoke Test

After services are up, run:

```bash
uv run cv-screener --help
```

If that works, the app code is installed and the CLI entrypoint is available.

## Run The App

Use the CLI for batch steps:

```bash
uv run cv-screener generate-cvs --count 3
uv run cv-screener ingest --reset
uv run cv-screener query "Who has Python experience?"
```

Use the UI for interactive chat:

```bash
uv run cv-screener serve
```

UI URL:

`http://127.0.0.1:8000/chainlit`
