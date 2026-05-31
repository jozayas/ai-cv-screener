# Architecture Notes

The repo is organized around a simple local-first CV screening flow.

## Main Parts

- `src/cv_screener/cv_generation`
  Generates structured fake CV content, optional synthetic photos, and rendered PDF CVs.

- `src/cv_screener/ingestion`
  Parses rendered PDFs, extracts structured text, chunks it, and prepares indexable records.

- `src/cv_screener/retrieval`
  Runs hybrid retrieval with semantic search in Qdrant and keyword search through BM25-style sparse vectors.

- `src/cv_screener/rag`
  Runs the question-answering graph, including routing, planning, retrieval, reranking, answer generation, and review.

- `src/cv_screener/chainlit`
  Thin UI layer for chat interactions.

- `src/cv_screener/cli`
  Thin command surface for the end-to-end workflow.

## Runtime Shape

At a high level:

`user question -> router -> planner -> retrieve -> rerank -> answer -> review -> final response`

The implementation is local-first:

- local files for generated inputs and outputs
- local SQLite for canonical persistence
- local Qdrant for retrieval
- local Ollama by default for OpenAI-compatible model access

## Scope Boundary

This repo currently supports the core local workflow:

- generate content
- render PDFs
- ingest documents
- query via CLI
- query via Chainlit UI

Anything beyond that should be documented from code that actually exists, not from older planning notes.
