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

- `OPENAI_BASE_URL=http://localhost:11434/v1`
- `OPENAI_API_KEY=ollama`
- `GENERATION_MODEL=gemma3:12b`
- `RAG_MODEL=gemma3:12b`
- `QDRANT_URL=http://localhost:6333`

If you want to use a different OpenAI-compatible endpoint, update `.env` and skip the local Ollama dependency.

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
