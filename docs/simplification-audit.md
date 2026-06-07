# Simplification Audit

## Scope and assumptions

- Scope: whole-package audit of `src/cv_screener`.
- Goal: make entrypoints explicit, reduce hidden control flow, remove workaround-driven abstractions, and centralize configuration.
- Assumption for the conflicting brief: "forget about performance for now" takes precedence over "improve performance". Performance is treated only as a secondary benefit when simplification also removes unnecessary work.
- This is an audit and plan document. It does not change behavior yet.
- This revision cross-checks the plan against the official documentation for the main libraries that shape architecture and runtime behavior.

## Documentation-backed constraints

The revised plan should follow these documented idioms as closely as possible:

- **Typer**
  - Use `@app.callback()` for app-level options and `typer.Context` for command context, instead of hidden state. Typer documents callback-level CLI parameters and direct context access.
    Sources:
    - https://typer.tiangolo.com/tutorial/commands/callback/
    - https://typer.tiangolo.com/tutorial/commands/context/
- **Pydantic Settings**
  - Use `BaseSettings` for environment-backed config.
  - Prefer nested settings models with `env_nested_delimiter` when configuration is hierarchical.
  - Let constructor overrides win when tests or commands need to override settings explicitly.
    Sources:
    - https://docs.pydantic.dev/latest/concepts/pydantic_settings/
    - https://docs.pydantic.dev/latest/api/pydantic_settings/
- **LangGraph**
  - Prefer `TypedDict` state as the main documented schema style.
  - Keep nodes as plain Python functions; LangGraph documents node functions directly and exposes `stream` / `astream`.
  - Runtime context belongs in the node/runtime interface, not in ad hoc globals.
    Sources:
    - https://docs.langchain.com/oss/python/langgraph/graph-api
    - https://docs.langchain.com/oss/python/langgraph/streaming
    - https://docs.langchain.com/oss/python/langgraph/use-graph-api
- **Chainlit**
  - Use `cl.user_session` for per-chat state, not process globals.
  - Use `Message` / `Step` streaming directly for chat progress.
    Sources:
    - https://docs.chainlit.io/concepts/user-session
    - https://docs.chainlit.io/advanced-features/streaming
- **Qdrant**
  - Use one long-lived `QdrantClient` per process composition root.
  - Use documented collection management and `query_points(..., prefetch=[...], query=FusionQuery(...))` for hybrid retrieval.
  - Named vectors and sparse vectors are first-class features, so the code should express them directly instead of wrapping them in workaround abstractions.
    Sources:
    - https://qdrant.tech/documentation/manage-data/collections/
    - https://api.qdrant.tech/api-reference/search/query-points/
    - https://qdrant.tech/documentation/search/hybrid-queries/
- **SQLAlchemy**
  - Create the `Engine` once and reuse it for the application lifetime.
  - For Core-oriented code, explicit engine/connection transaction scopes are idiomatic.
  - If moving to ORM-style persistence later, use `sessionmaker` as a module-level factory.
    Sources:
    - https://docs.sqlalchemy.org/en/14/core/connections.html
    - https://docs.sqlalchemy.org/en/20/orm/session_basics.html
- **PyMuPDF4LLM**
  - Prefer supported function parameters like `page_chunks=True` and `show_progress=False` over indirect output-suppression workarounds.
    Source:
    - https://pymupdf.readthedocs.io/en/latest/pymupdf4llm/api.html
- **Sentence Transformers CrossEncoder**
  - Use the documented `CrossEncoder(...).predict(...)` surface directly.
  - Keep the reranker implementation aligned with the library’s concrete API rather than hiding it behind extra local strategy layers.
    Source:
    - https://sbert.net/docs/package_reference/cross_encoder/model.html

## Current entrypoints

- CLI root: `cv_screener:main` via [pyproject.toml](/home/jozayas/Leadtech/cv-screener/pyproject.toml:33)
- Typer app shell: [src/cv_screener/cli/app.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/cli/app.py:1)
- Registered CLI commands: [src/cv_screener/cli/commands.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/cli/commands.py:32)
- Chainlit host launcher: [src/cv_screener/cli/serve.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/cli/serve.py:19)
- Chainlit chat handler: [src/cv_screener/chainlit/app.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/chainlit/app.py:115)
- RAG application service: [src/cv_screener/rag/service.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/rag/service.py:50)
- Ingestion application service: [src/cv_screener/ingestion/ingest.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/ingestion/ingest.py:61)

## Findings

### 1. CLI entrypoints are obscured by builder indirection and runtime globals

**Where**

- [src/cv_screener/cli/dependencies.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/cli/dependencies.py:20)
- [src/cv_screener/cli/runtime.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/cli/runtime.py:43)
- [src/cv_screener/cli/generation.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/cli/generation.py:20)
- [src/cv_screener/cli/commands.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/cli/commands.py:72)

**What is overengineered**

- `cli/dependencies.py` defines multiple local `Protocol` interfaces plus `import_module` factories and `cast(...)` wrappers just to instantiate concrete services.
- `cli/runtime.py` keeps both `ctx.obj` and a mutable module-global `_runtime_state`, so command behavior depends on hidden state when `ctx=None`.
- `generate_cvs()` bypasses normal CLI context by passing `ctx=None`, which depends on the runtime global to behave correctly.

**Why it hurts**

- The real command flow is hard to trace.
- Behavior differs between "called from a Typer command" and "called as an internal helper".
- The abstraction mainly exists to keep import-time work low, but the price is hidden wiring everywhere.

**Suggested fix**

- Replace `cli/dependencies.py` with one explicit composition module that returns concrete services, not local protocols.
- Remove `_runtime_state`; require a resolved runtime object to be passed explicitly to helper functions.
- Split orchestration from command handlers:
  - command parses args/options
  - command builds settings/services
  - command calls a plainly named application function
- Make `generate_cvs()` a normal orchestrator that explicitly initializes runtime instead of relying on `ctx=None`.
- This matches Typer’s documented callback/context model more closely than the current hybrid of `ctx.obj` plus module-global runtime state.

### 2. Configuration is fragmented and drifts between env, CLI defaults, and hard-coded values

**Where**

- [src/cv_screener/config.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/config.py:17)
- [src/cv_screener/retrieval/schema.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/retrieval/schema.py:8)
- [src/cv_screener/ingestion/indexing/schema.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/ingestion/indexing/schema.py:10)
- [src/cv_screener/ingestion/chunking/schema.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/ingestion/chunking/schema.py:61)
- [src/cv_screener/cli/commands.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/cli/commands.py:53)
- [src/cv_screener/cli/dependencies.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/cli/dependencies.py:111)
- [src/cv_screener/rag/service.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/rag/service.py:58)
- [src/cv_screener/persistence/lookup.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/persistence/lookup.py:22)

**What is overengineered**

- Some settings are `BaseSettings`, some are plain `BaseModel`, some are module constants, some are CLI defaults, and some are duplicated magic values.
- Example duplicates:
  - `candidate_name_min_score`: config setting and separate `_DEFAULT_MIN_NAME_MATCH_SCORE`
  - Qdrant URL comes from env, but collection names and model names live in non-env `BaseModel` config objects
  - CLI paths like `data/cvs_contents`, `data/cv_pdfs`, and `data/generated/photos` are hard-coded in commands and builders

**Why it hurts**

- The operator cannot reliably answer "where do I change this?"
- The codebase has several pseudo-source-of-truth layers.
- Hidden defaults make test behavior and CLI behavior diverge.

**Suggested fix**

- Create one top-level `AppSettings` tree using `BaseSettings`, with nested sections for:
  - paths
  - generation
  - photo generation
  - chunking
  - indexing
  - retrieval
  - rag
  - lookup
  - serve
- Standardize on nested environment variables that mirror that tree, instead of keeping both nested and flat aliases for the same setting.
- CLI options should override these settings explicitly.
- Non-CLI code should accept config objects, not construct new settings internally.
- Remove duplicated fallback constants once the setting exists in `AppSettings`.
- Use `env_nested_delimiter` so env configuration remains hierarchical and documented.

### 3. Retrieval and indexing repeat the same configuration concepts in separate models

**Where**

- [src/cv_screener/retrieval/schema.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/retrieval/schema.py:8)
- [src/cv_screener/ingestion/indexing/schema.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/ingestion/indexing/schema.py:10)
- [src/cv_screener/retrieval/hybrid.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/retrieval/hybrid.py:21)
- [src/cv_screener/ingestion/indexing/qdrant.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/ingestion/indexing/qdrant.py:29)

**What is overengineered**

- Retrieval and indexing each define their own copies of collection names, vector names, embedding model names, and BM25 model names.
- Both build their own `QdrantClient`, `TextEmbedding`, and `BM25Encoder` with nearly identical patterns.

**Why it hurts**

- Shared behavior is conceptually one subsystem but operationally split across two config models.
- Drift is likely, especially if the embedding model or collection shape changes.

**Suggested fix**

- Introduce one explicit retrieval/index settings object shared by both the indexer and retriever.
- Extract a single composition-level "vector backend" factory that owns:
  - qdrant client creation
  - embedding model creation
  - optional BM25 encoder creation
- Keep only one place where collection/vector naming lives.
- Express hybrid retrieval directly through documented Qdrant `prefetch` + fusion query construction.

### 4. Protocol usage is excessive and appears aimed at satisfying typing friction more than clarifying interfaces

**Where**

- [src/cv_screener/cli/dependencies.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/cli/dependencies.py:20)
- [src/cv_screener/ingestion/ingest.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/ingestion/ingest.py:21)
- [src/cv_screener/ingestion/indexing/protocols.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/ingestion/indexing/protocols.py:16)
- [src/cv_screener/rag/graph.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/rag/graph.py:49)
- [src/cv_screener/rag/nodes/retrieve.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/rag/nodes/retrieve.py:14)
- [src/cv_screener/rag/nodes/rerank.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/rag/nodes/rerank.py:23)
- [src/cv_screener/rag/nodes/targeted_lookup.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/rag/nodes/targeted_lookup.py:16)
- [src/cv_screener/cv_generation/content/generator.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/cv_generation/content/generator.py:30)
- [src/cv_screener/cv_generation/content/sources.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/cv_generation/content/sources.py:28)

**What is overengineered**

- The same conceptual interfaces are defined more than once.
- Several protocols are used only to support lazy imports and casts.
- `CompiledRAGGraph` is a local protocol wrapping a concrete compiled graph object.

**Why it hurts**

- The code looks more abstract than it really is.
- It is harder to tell which interfaces are stable contracts and which are local shims.
- It increases type noise without making entrypoints clearer.

**Suggested fix**

- Keep protocols only where they are genuine seams for testing or alternate implementations.
- Prefer concrete types for internal composition.
- If a dependency is only swapped in tests, use a narrow callable/object contract in the constructor without duplicating that contract in multiple modules.
- Remove duplicate `CVProfileSource` protocol declarations and keep one shared definition.
- Prefer documented library types at the composition root instead of wrapping them just to satisfy type-checking friction.

### 5. The RAG graph is wrapped in repetitive adapter functions and state patching

**Where**

- [src/cv_screener/rag/graph.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/rag/graph.py:83)

**What is overengineered**

- The file is dominated by near-identical wrappers that:
  - call a node function
  - pass `get_config()` and one dependency
  - append `nodes_executed`
  - cast the result

**Why it hurts**

- It hides the actual graph topology behind ceremony.
- Changes to one node signature cascade into boilerplate edits.
- `nodes_executed` tracking is spread across wrappers rather than being a graph concern.

**Suggested fix**

- Keep `TypedDict` state and plain node functions, as LangGraph documents.
- Simplify wrappers so the graph file focuses on topology and dependency injection.
- Move execution-trace updates into one small helper if they remain necessary.
- Remove `cast("dict[str, object]", ...)` noise wherever node functions can return typed updates directly.
- In doubt, prefer LangGraph’s direct node-function shape over custom wrapper abstractions.

### 6. Targeted lookup is an ad hoc rule engine buried inside a graph node

**Where**

- [src/cv_screener/rag/nodes/targeted_lookup.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/rag/nodes/targeted_lookup.py:164)
- [src/cv_screener/rag/graph.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/rag/graph.py:325)
- [src/cv_screener/rag/nodes/finalize.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/rag/nodes/finalize.py:77)

**What is overengineered**

- Regex routing, skill-term blocking, clarification behavior, semantic fallback, "profile" mode, and list formatting are distributed across three modules.
- `response_mode` and `fallback_to_semantic` encode control flow as stringly state rather than explicit branching decisions.
- `full_cv_to_chunks_node()` manufactures a fake retrieved chunk with section `"FULL_CV"` to reuse the answer path.

**Why it hurts**

- The control flow is not obvious from the entrypoint.
- Small wording changes in queries can affect routing in hard-to-debug ways.
- The graph path is partly domain logic and partly workaround logic.

**Suggested fix**

- Pull targeted lookup into a dedicated application service with explicit result types such as:
  - `ClarificationNeeded`
  - `CandidateList`
  - `DirectProfile`
  - `FallBackToRetrieval`
- Let the graph branch on those result types directly.
- Remove the "fake chunk" workaround and give the answer step an explicit full-document context path if that behavior is still needed.
- Consider using a documented LangGraph `Command` return only if it genuinely simplifies update-and-route behavior; do not introduce it unless it clearly reduces state indirection.

### 7. Chainlit conversation handling duplicates RAG routing concerns with UI-side query rewriting

**Where**

- [src/cv_screener/chainlit/app.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/chainlit/app.py:55)
- [src/cv_screener/chainlit/ui.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/chainlit/ui.py:69)

**What is overengineered**

- The UI layer rewrites queries for ordinals and pronouns before they reach the router.
- Conversation context is hand-built as plain text snippets in the UI session.
- The UI builds and caches its own service through the same builder indirection as the CLI.

**Why it hurts**

- Query understanding lives partly in UI and partly in RAG runtime.
- It is harder to reason about behavior from `query` vs `Chainlit`.
- Session behavior is hidden in Chainlit callbacks instead of reusable application code.

**Suggested fix**

- Move candidate-reference resolution into the RAG application layer.
- Keep Chainlit as a thin adapter that forwards message text, uses `user_session` for per-chat state, and renders streamed `Message` / `Step` updates.
- Replace ad hoc session string assembly with a small conversation state object owned by the RAG service or a dedicated chat session helper.

### 8. Ingestion orchestration contains compatibility fallbacks and progress math inside the main service path

**Where**

- [src/cv_screener/ingestion/ingest.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/ingestion/ingest.py:80)

**What is overengineered**

- The service handles parsing, chunking, persistence, indexing, progress stage calculation, and parser compatibility fallback.
- `_parse_cvs()` catches `TypeError` to guess whether the parser supports a progress callback.

**Why it hurts**

- Compatibility logic obscures the main ingestion flow.
- The `TypeError` fallback is a workaround, not a contract.

**Suggested fix**

- Make parser progress support explicit in one interface.
- Move progress reporting out of the business service into a wrapper or reporter object.
- Keep `CVIngestionService.ingest()` as a plain sequence:
  - parse
  - chunk
  - persist
  - index
  - summarize
- Replace environment and warning workarounds with documented library parameters where available.

### 9. Library-noise suppression is scattered and hidden in runtime paths

**Where**

- [src/cv_screener/ingestion/parser.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/ingestion/parser.py:62)
- [src/cv_screener/ingestion/chunking/sectioning.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/ingestion/chunking/sectioning.py:22)

**What is overengineered**

- Environment mutations, warning filters, and stdout/stderr redirection happen inside parsing and section-classification code.
- These are runtime workarounds for third-party noise, but they are embedded in domain logic.

**Why it hurts**

- Hidden side effects make behavior harder to predict.
- The same concern is solved differently in multiple places.

**Suggested fix**

- Centralize external-library noise suppression in one runtime/bootstrap module.
- Keep parsing/chunking code free of environment mutation.
- For PyMuPDF4LLM specifically, prefer supported parameters such as `show_progress=False` before mutating process environment.
- If a library requires a one-time quiet setup, do it once during service construction.

### 10. RAG structured-output handling contains defensive repair logic that broadens behavior too much

**Where**

- [src/cv_screener/rag/llm.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/rag/llm.py:68)

**What is overengineered**

- The LLM layer augments prompts, parses multiple response shapes, heuristically extracts JSON from mixed text, and retries with a repair prompt.
- This is robust, but it also turns one clear contract into a permissive recovery system.

**Why it hurts**

- Node behavior is harder to reason about because the model layer silently repairs malformed output.
- Failures that should be visible are partially normalized.

**Suggested fix**

- Decide on one strict structured-output path.
- Keep one small fallback at most, and log when it is used.
- Avoid generic "accept anything that looks like JSON" behavior unless it is clearly required.
- If documentation does not support a recovery path, treat it as suspicious and keep it explicit.

### 11. Reviewer and reranker keep more strategy surface than the current product needs

**Where**

- [src/cv_screener/rag/nodes/review.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/rag/nodes/review.py:28)
- [src/cv_screener/rag/nodes/rerank.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/rag/nodes/rerank.py:17)

**What is overengineered**

- Reviewer logic keeps a configurable revision loop but runtime defaults are effectively fixed.
- Reranker keeps multiple model constants and strategy classes while the real branch is just "cross-encoder on/off".

**Why it hurts**

- The code advertises more runtime flexibility than the CLI/config surface exposes.
- It increases the amount of code a reader must load to understand the actual behavior.

**Suggested fix**

- Move the real runtime knobs into config and CLI if they are intended to vary.
- Otherwise collapse unused strategy surface:
  - one configured reranker implementation
  - one explicit review policy object or function
- Keep the reranker aligned with the documented `CrossEncoder.predict(...)` API.

### 12. Persistence code mixes canonical storage with heuristic extraction and legacy migration

**Where**

- [src/cv_screener/persistence/repository.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/persistence/repository.py:34)
- [src/cv_screener/persistence/lookup.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/persistence/lookup.py:22)

**What is overengineered**

- Repository code owns schema creation, migration/backfill, candidate ID inference, skill parsing heuristics, education parsing heuristics, and row replacement.
- Lookup code duplicates normalization concerns and owns fuzzy thresholds separately.

**Why it hurts**

- Canonical persistence is not clearly separated from extraction heuristics.
- The storage layer has business rules mixed into it.

**Suggested fix**

- Split persistence responsibilities:
  - schema/bootstrap
  - canonical writes
  - extracted metadata heuristics
  - lookup/search helpers
- Introduce a generic database configuration and connection factory at the SQLAlchemy layer, using a database URL instead of SQLite-specific construction in application services.
- Keep SQLite as the default local deployment target, but make repositories depend on a generic SQLAlchemy `Engine` / session boundary so PostgreSQL or another backend can replace it without touching application logic.
- Move thresholds into shared config.
- If legacy migration is still needed, isolate it behind an explicit migration step instead of running it as part of normal persistence.

### 13. Duplicate source abstractions exist in CV generation

**Where**

- [src/cv_screener/cv_generation/content/generator.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/cv_generation/content/generator.py:30)
- [src/cv_screener/cv_generation/content/sources.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/cv_generation/content/sources.py:28)
- [src/cv_screener/cli/generation.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/cli/generation.py:54)

**What is overengineered**

- `CVProfileSource` is declared twice.
- `build_profile_source()` uses dynamic imports and casts to construct two concrete implementations that are already known in advance.

**Why it hurts**

- The generation flow is more abstract than the problem requires.

**Suggested fix**

- Keep one source protocol or abstract base.
- Build the source explicitly in one composition function using real imports.

### 14. Configuration still duplicates the same fields across nested app config and flat compatibility settings

**Where**

- [src/cv_screener/config.py](/home/jozayas/Leadtech/cv-screener/src/cv_screener/config.py:1)

**What is overengineered**

- The file currently keeps:
  - one nested `AppSettings` tree for composition
  - multiple flat `BaseSettings` classes for compatibility
- This preserves behavior, but it duplicates the same generation, RAG, Qdrant, SQLite, and lookup fields in two places.

**Why it hurts**

- It creates two maintenance surfaces for the same defaults and validation rules.
- The file is simpler than the previous mixin-based version, but still not at the target level of clarity.

**Suggested fix**

- Collapse toward one primary settings representation.
- Prefer one of these end states:
  - `AppSettings` as the only source of truth, with consumers migrated off flat settings classes
  - or one small set of flat `BaseSettings` classes plus a very thin `AppSettings` facade that only groups them without repeating fields
- Do not keep both duplicated long-term.

## Priority plan

### Phase 1: make configuration have one documented source of truth

1. Introduce one nested `AppSettings` tree using `BaseSettings` and `env_nested_delimiter`.
2. Move hard-coded paths, thresholds, and model names into that tree.
3. Make CLI options override settings explicitly at the command boundary.
4. Remove duplicated defaults from services and lookup modules.
5. Document every supported env variable in one place.

### Phase 2: align CLI composition with Typer idioms

1. Replace `cli/dependencies.py` with one concrete composition module.
2. Remove runtime globals from `cli/runtime.py`.
3. Keep app-level options in the Typer callback and pass resolved context/settings explicitly.
4. Make every command initialize runtime and dependencies explicitly.
5. Ensure CLI and Chainlit both create services through the same composition path.

### Phase 3: collapse unnecessary protocols, casts, and lazy wrappers

1. Delete local protocols that only exist for lazy import wrappers.
2. Keep only genuine seams used by tests or alternate implementations.
3. Replace dynamic-import composition with direct documented imports at the composition root.
4. Remove pyrefly-oriented casts once composition uses concrete types.

### Phase 4: simplify the RAG runtime in documented LangGraph style

1. Keep `TypedDict` state and plain node functions.
2. Refactor `rag/graph.py` so the file shows topology first and minimal helper glue second.
3. Extract targeted lookup into an explicit service with typed outcomes.
4. Remove fake-retrieved-chunk bridging for full CV/profile handling.
5. Move candidate-reference rewriting out of Chainlit into the RAG application layer.
6. Only adopt `Command`-based node routing if it clearly simplifies update-and-route logic.

### Phase 5: replace workaround flow with documented library behavior

1. Remove `TypeError`-based parser capability fallback in ingestion.
2. Replace output/progress workarounds with documented library options where available.
3. Centralize any remaining library noise suppression for Hugging Face and related dependencies.
4. Isolate LLM structured-output repair behavior behind one narrow, explicit policy.

### Phase 6: reduce strategy surface to what is actually supported and configured

1. Collapse reranker configuration and implementation surface.
2. Collapse reviewer loop behavior to one explicit policy.
3. Expose any remaining runtime variability through config/CLI, or delete it.

### Phase 7: align persistence and infrastructure factories with SQLAlchemy and Qdrant idioms

1. Replace SQLite-specific connection construction with a generic SQLAlchemy database URL setting and one composition-level `Engine` factory.
2. Make repositories depend on a generic SQLAlchemy database boundary instead of directly on SQLite path construction.
3. Keep SQLite as the default local backend, but make backend substitution a configuration concern instead of an application-code concern.
4. Create long-lived composition-level factories for `Engine` and `QdrantClient`.
5. Split repository bootstrap/migration from canonical writes.
6. Move extraction heuristics to dedicated helper modules.
7. Move lookup thresholds and normalization policy into shared configuration/utilities.

### Phase 8: remove duplicated config representations

1. Choose one primary configuration model.
2. Migrate remaining consumers so field definitions live in one place.
3. Delete the duplicate settings/config layer once all callers are switched.

## Recommended execution order

1. Config unification
2. CLI/runtime composition cleanup
3. Protocol/cast removal
4. Ingestion flow cleanup
5. Retrieval/index config unification
6. RAG graph and targeted lookup simplification
7. Chainlit thin-adapter cleanup
8. Persistence separation with generic SQLAlchemy database boundary
9. Config representation deduplication
10. LLM/reviewer/reranker surface reduction

## Confirmed implementation decisions

These decisions are now fixed unless later evidence shows a materially simpler path:

1. **SQLAlchemy rewrite scope**
   - A bigger rewrite is allowed when it produces a significant simplification benefit.
   - Default direction: keep the current persistence style close to the existing Core-oriented flow, but do not preserve it out of inertia.
   - If switching part of the persistence boundary to a `sessionmaker`-based ORM shape substantially reduces repository complexity, duplication, or transaction boilerplate, that rewrite is in scope.
   - The decision standard is simplification first, not feature expansion.

2. **LangGraph routing shape**
   - Use the simpler documented LangGraph shape.
   - Conditional edges do not need to be preserved if `Command` produces a clearer update-and-route flow.
   - Since you approved the second point, the implementation may adopt the more direct documented mechanism where it reduces wrapper logic.

3. **LLM JSON strictness**
   - Move to a stricter structured-output path.
   - Do not keep broad repair heuristics or suspicious fallback parsing unless a concrete documented need remains.
   - Failures should become more visible rather than being normalized by hidden workaround logic.

## Expected outcomes

- Every runtime knob can be found in one settings tree and overridden from env or CLI.
- Command entrypoints read top-to-bottom without dynamic import hunting.
- RAG routing decisions are explicit instead of encoded through string modes and fallback flags.
- UI behavior no longer rewrites domain queries behind the graph's back.
- Workarounds for third-party noise and malformed LLM output are isolated and visible.
- The package becomes easier to review, debug, and change without changing product behavior.
