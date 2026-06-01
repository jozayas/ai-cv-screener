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

```txt
user question
  -> router
     -> brief_answer
     -> return_cv
     -> targeted_lookup
     -> planner
```

The router now has four meaningful runtime branches:

1. `small_talk` / `needs_clarification`
   - `router -> brief_answer -> finalize`

2. Explicit CV requests
   - examples: `Give me the CV of Alejandro García Martínez`
   - path: `router -> return_cv -> finalize`
   - behavior: bypass retrieval and return the resolved CV document directly

3. Deterministic targeted lookup requests
   - examples: `Who knows Python?`, `Who has Python experience?`, `Summarize the profile of Alejandro García Martínez`
   - path starts at `router -> targeted_lookup`
   - three sub-paths exist:
     - clarification: `targeted_lookup -> finalize`
     - candidate list: `targeted_lookup -> hydrate -> finalize`
     - named profile summary: `targeted_lookup -> return_profile_cv -> profile_context -> answer -> review -> finalize`

4. General recruiter-style CV queries
   - examples: `Who has Python backend experience?`
   - path: `router -> planner -> retrieve -> rerank -> answer -> review -> finalize`

## Deterministic Lookup Behavior

The targeted lookup layer is SQLite-first and now behaves as follows:

- named profile requests do not fall back to semantic retrieval
- a single matched candidate resolves to the candidate CV document first
- zero matches returns a clarification prompt asking for the full name
- multiple matches returns a clarification prompt with the matched full names
- simple skill queries return every matching candidate once, with deduped evidence in the final response

## Profile Summary Path

Named profile summaries now use the candidate CV as the source of truth:

```txt
Summarize the profile of <candidate>
  -> targeted_lookup
  -> return_profile_cv
  -> profile_context
  -> answer
  -> review
  -> finalize
```

Implementation detail:

- `return_profile_cv` resolves the matched document through the same canonical lookup used by direct CV requests
- `profile_context` converts the resolved parsed CV markdown into a single answer-context chunk
- this path skips hybrid retrieval and reranking entirely

This change was introduced because the prior targeted profile path depended on chunk hydration and could abstain even when direct document lookup already worked.

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
