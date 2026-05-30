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
* LangChain + LangGraph for RAG orchestration
* Qdrant for vector search
* BM25 for keyword search
* Cross-encoder reranking before answer generation
* OpenAI-compatible model interface
* Ollama/local models by default
* Chainlit for the UI
* LangSmith for optional tracing/evaluation
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

Avoid unnecessary abstractions, auth, background workers, Kubernetes, and custom frontend work unless explicitly requested.

For the RAG runtime, prefer a small, explicit LangGraph multi-agent workflow over a generic autonomous agent system.

Before ignoring, suppressing, or weakening any lint rule or typing rule, ask the user what to do.

Always use the `using-agent-skills` skill at the beginning of a session.

## Library Idioms

Follow the idiomatic usage of the libraries in this stack:

* Typer: keep CLI commands thin, use typed `Annotated` arguments/options, and delegate business logic to services.
* Pydantic: validate data at boundaries, use models for structured contracts, `BaseSettings` for environment config, and `SecretStr` for secrets.
* YAML: keep generated CV YAML as plain source data and validate it through Pydantic instead of relying on ad hoc parsing.
* Jinja2 + WeasyPrint: keep presentation in templates, keep rendering deterministic, and keep data shaping in Python.
* PyMuPDF: parse PDFs into explicit intermediate models before chunking, retrieval, or evaluation logic.
* LangChain + LangGraph: prefer plain node functions over callable classes unless stateful encapsulation is clearly needed; prefer `TypedDict` state or `MessagesState` for chat flows; keep deterministic orchestration outside the LLM and use structured outputs for control nodes.
* Qdrant + BM25 + reranking: keep retrieval deterministic, configuration explicit, and index/query contracts strongly typed.
* Chainlit: keep the UI layer thin; callbacks should call application services or graph entrypoints rather than holding business logic.
* LangSmith: keep tracing optional and non-blocking; do not make the main local workflow depend on LangSmith being configured.
* Docker Compose: keep local infrastructure minimal, explicit, and aligned with the development workflow rather than production orchestration.

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

## RAG Runtime Architecture

For the RAG application layer, use a runtime graph with specialist nodes:

```txt
user message
  → router
  → planner / reformulator
  → hybrid retrieval
  → reranking
  → grounded answer generation
  → groundedness review
  → Chainlit response with sources
```

Guidelines:

* Use LangGraph to model the runtime workflow.
* Prefer plain LangGraph node functions over callable classes unless stateful encapsulation is clearly needed.
* Prefer `TypedDict` state, or `MessagesState` for chat-style flows; use Pydantic state only when runtime validation of state updates is needed.
* Keep retrieval deterministic and reuse the existing hybrid retrieval backend.
* Use structured outputs for router, planner, answer, and review nodes.
* Route greetings and trivial chat away from retrieval when possible.
* Allow query reformulation to improve recall on recruiter-style questions.
* Allow at most one groundedness revision pass before finalizing or abstaining.
* Do not build a broad autonomous supervisor or tool-using agent system.

## CLI

Use Typer. Target commands:

```bash
uv run cv-screener generate-content --count 30
uv run cv-screener generate-cvs --count 30
uv run cv-screener validate data/cvs_contents
uv run cv-screener render data/cvs_contents
uv run cv-screener ingest --reset
uv run cv-screener query "Who has Python experience?"
uv run cv-screener eval-retrieval
uv run cv-screener eval-rag
uv run cv-screener serve
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
  → optional query reformulation
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

Suggested runtime defaults:

```txt
rewrite_count = 3
review_passes = 1
```

## Answering Rules

The RAG system must:

* answer only from retrieved CV context
* avoid inventing candidates, skills, education, or experience
* say when the CVs do not contain enough information
* include source citations in every answer
* prefer concise answers
* run a groundedness review before returning substantive answers

The router should be able to:

* reply directly to greetings or trivial non-CV turns
* ask for clarification when a user request is too vague to retrieve well

## Evaluation

Use the generated YAML profiles as ground truth.

Implement:

* retrieval evaluation: Recall@5, MRR, Precision@5, source hit rate
* RAG evaluation: expected candidate present, sources included, unsupported claims avoided

Prefer deterministic checks before LLM-as-judge.

LangSmith tracing/evaluation is optional during implementation and should be added only after the main RAG flow works end to end.

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
12. LangGraph RAG runtime:
    * router
    * planner / reformulator
    * grounded answerer
    * groundedness reviewer
13. CLI query command
14. Chainlit UI
15. Evaluation
16. LangSmith tracing/evaluation
17. Docker Compose
18. README/demo instructions

Start with 3 CVs. Scale to 25–30 only after the full pipeline works.
