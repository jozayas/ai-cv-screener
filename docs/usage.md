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
