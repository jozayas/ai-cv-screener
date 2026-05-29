# AGENTS.md

## Project

Build a local-first AI-powered CV screener for a technical challenge.

The app must generate fake CV PDFs, ingest them into a RAG pipeline, and provide a chat UI for asking questions about the candidates. The system should answer only from the CV contents and include source citations.

## Stack

Use:

* Python
* Typer for CLI
* Pydantic for validation
* YAML for generated CV profiles
* Jinja2 + WeasyPrint for PDF generation
* PyMuPDF for PDF parsing
* LangChain for RAG
* Qdrant for vector search
* BM25 for keyword search
* Reranking before answer generation
* OpenAI-compatible model interface
* Ollama/local models by default
* Chainlit for the UI
* Docker Compose for local deployment

## Development Rules

Work feature by feature. Do not implement the whole project at once.

For each requested feature:

1. Make the smallest useful change.
2. Keep code simple, typed, and readable.
3. Add or update validation where relevant.
4. Add or update CLI commands where useful.
5. Add minimal tests or manual verification steps.
6. Stop when the requested feature is complete.

Avoid unnecessary abstractions, agents, auth, background workers, Kubernetes, and custom frontend work unless explicitly requested.

## Required Architecture

```txt
YAML CV generation
  → Pydantic validation
  → PDF rendering
  → PDF parsing
  → section-aware + semantic chunking
  → Qdrant vector index + BM25 keyword index
  → hybrid retrieval
  → reranking
  → grounded RAG answer
  → Chainlit UI with sources
```

## CLI

Use Typer. Target commands:

```bash
python -m app.cli generate-cvs --count 30
python -m app.cli validate-cvs
python -m app.cli render-pdfs
python -m app.cli ingest --reset
python -m app.cli query "Who has Python experience?"
python -m app.cli eval-retrieval
python -m app.cli eval-rag
python -m app.cli serve
```

## Core Requirements

* Generate 25–30 realistic fake CVs as PDFs.
* Keep YAML CV profiles as source data and evaluation ground truth.
* Parse the generated PDFs with PyMuPDF.
* Use parsed PDF text for RAG, not the YAML directly.
* Implement hybrid retrieval: Qdrant semantic search + BM25 + fusion.
* Implement reranking before final context selection.
* Generate grounded answers using an OpenAI-compatible model client.
* Include source filenames, pages, or sections in answers.
* Add retrieval and RAG evaluation.
* Provide Docker Compose and clear local setup instructions.

## Retrieval Flow

```txt
user query
  → semantic search
  → BM25 search
  → fusion
  → reranking
  → final context
  → answer with sources
```

Suggested defaults:

```txt
semantic_top_k = 12
bm25_top_k = 12
fusion_top_k = 10
rerank_top_k = 5
```

## Answering Rules

The RAG system must:

* answer only from retrieved CV context
* avoid inventing candidates, skills, education, or experience
* say when the CVs do not contain enough information
* include source citations in every answer
* prefer concise answers

## Evaluation

Use the generated YAML profiles as ground truth.

Implement:

* retrieval evaluation: Recall@5, MRR, Precision@5, source hit rate
* RAG evaluation: expected candidate present, sources included, unsupported claims avoided

Prefer deterministic checks before LLM-as-judge.

## Implementation Order

1. Project scaffold
2. Pydantic CV schema
3. YAML validation
4. LLM-based YAML CV generation
5. PDF rendering
6. PDF parsing
7. Chunking
8. Qdrant indexing
9. BM25 indexing
10. Hybrid retrieval
11. Reranking
12. RAG answering
13. CLI query command
14. Chainlit UI
15. Evaluation
16. Docker Compose
17. README/demo instructions

Start with 3 CVs. Scale to 25–30 only after the full pipeline works.
