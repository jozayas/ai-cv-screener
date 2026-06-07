# Usage

This repo works as a small local pipeline:

`YAML profiles -> PDF CVs -> parsed chunks -> indexed retrieval -> RAG answers`

## Standard Workflow

1. Generate source CV content:

```bash
uv run cv-screener generate-content --count 3
```

2. Or generate content, photos, and PDFs together:

```bash
uv run cv-screener generate-cvs --count 3
```

3. Validate YAML inputs:

```bash
uv run cv-screener validate data/cvs_contents
```

4. Render PDFs from YAML:

```bash
uv run cv-screener render data/cvs_contents
```

5. Parse and index the rendered PDFs:

```bash
uv run cv-screener ingest --reset
```

6. Ask questions from the CLI:

```bash
uv run cv-screener query "Which candidates mention Python and FastAPI?"
uv run cv-screener query "Who knows Python?"
uv run cv-screener query "Summarize the profile of Alejandro García Martínez"
```

7. Run the Chainlit chat UI:

```bash
uv run cv-screener serve
```

## Data Locations

- YAML CV content: `data/cvs_contents`
- Rendered PDFs: `data/cv_pdfs`
- Generated photos: `data/generated/photos`
- Local SQLite DB: `data/cv_screener.db`

## Current CLI Surface

The repo currently exposes these commands:

- `generate-content`
- `generate-cvs`
- `generate-photos`
- `validate`
- `render`
- `ingest`
- `query`
- `serve`

Older documentation that references additional commands is stale.

## Query Behavior Notes

- `Give me the CV of <candidate>` returns the matched CV document directly.
- `Summarize the profile of <candidate>` resolves the matched CV document directly, then summarizes from that CV content.
- `Summarize the profile of <partial name>` asks for clarification when the name is missing or ambiguous.
- `Who knows Python?` and `Who has Python experience?` are routed to the targeted lookup branch when the router classifies them as exact skill lookups.
- Broader recruiter-style questions like `Who has Python backend experience?` still use the planner/retrieval/rerank path.
