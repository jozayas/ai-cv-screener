# Day 1

## Scope completed

* Project scaffold and CLI baseline
* YAML CV content generation
* Pydantic schema validation
* Seeded fallback generation path
* OpenAI-compatible LLM generation path
* Jinja2 + WeasyPrint PDF rendering
* Initial PDF template set with deterministic template selection
* CV generation module refactor into content and PDF subpackages
* CLI/logging/progress cleanup
* Docker Compose setup for Ollama

## Key decisions

* YAML is the source of truth for CV content
* `candidate_id` is UUID-based
* output filenames use `slugified-name-uuid`
* canonical identity lives in YAML, not in the filename
* logs and progress go to `stderr`
* final CLI output goes to `stdout`
* default local model is `gemma3:12b`
* Ollama image is pinned to `ollama/ollama:0.24.0`
* PDF template and section-order selection is deterministic per CV

## Refactors made

* Split content generation from PDF rendering inside `cv_generation`
* Split PDF rendering into renderer, template metadata, and section payload helpers
* Simplified PDF template metadata so visible section titles live with each template definition

## Validation done

* `uv run ruff check .`
* `uv run pytest tests/test_cv_schema.py tests/test_pdf_rendering.py`
* `uv run pyrefly check`
* local PDF rendering was exercised against generated YAML profiles

## Notes

* Ollama GPU access was debugged and verified on CUDA
* `qwen3:8b` was unreliable for strict JSON generation
* live validation of the `gemma3:12b` generation path still needs dedicated follow-up
* PDF parsing, ingestion, retrieval, reranking, RAG answering, and UI are still pending
