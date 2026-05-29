# Architecture

## Overview

This project is a local-first AI-powered CV screener.

It generates fake CVs, renders them as PDFs, parses those PDFs, indexes their contents, and allows users to ask grounded questions about the candidates through a chat UI.

## Pipeline

```txt
YAML CV generation
  → Pydantic validation
  → HTML template rendering
  → PDF generation
  → PDF parsing
  → section-aware chunking
  → semantic chunking
  → Qdrant vector indexing
  → BM25 keyword indexing
  → hybrid retrieval
  → reranking
  → grounded answer generation
  → Chainlit UI
```

## Main Components

### 1. CV Generation

CVs are generated as YAML files first.

The YAML files are the structured source of truth for:

* validation
* PDF rendering
* evaluation ground truth
* reproducibility

The generator uses an OpenAI-compatible model interface, so the same code path can work with:

* Ollama
* OpenRouter
* Gemini through an OpenAI-compatible proxy
* OpenAI-compatible local or hosted models

### 2. PDF Rendering

Validated YAML profiles are rendered into HTML using Jinja2 templates.

The HTML is then exported to PDF.

The rendered PDFs are the documents ingested by the RAG pipeline.

### 3. PDF Parsing

PDFs are parsed with PyMuPDF.

The parser extracts:

* text
* page numbers
* candidate filename
* detected sections where possible

The RAG pipeline uses parsed PDF text, not the YAML files directly. This keeps the document-processing workflow realistic.

### 4. Chunking

Chunking is section-aware first.

For example:

* Summary
* Experience
* Skills
* Education
* Languages
* Projects
* Certifications

Large sections may then be split using semantic chunking.

This preserves CV structure while keeping chunks small enough for accurate retrieval.

### 5. Indexing

The system builds two indexes:

```txt
Qdrant index:
  semantic vector search

BM25 index:
  keyword and exact-term search
```

Each chunk stores metadata:

* candidate name
* source filename
* page number
* section
* chunk id
* source text

### 6. Hybrid Retrieval

Hybrid retrieval is mandatory.

The retrieval flow is:

```txt
user query
  → semantic search in Qdrant
  → BM25 keyword search
  → fusion
  → reranking
  → final context
```

Semantic search is useful for broad questions such as:

```txt
Who is the best candidate for an AI backend role?
```

BM25 is useful for exact terms such as:

```txt
Python
Kubernetes
German
UPC
AWS
FastAPI
```

Fusion combines both result sets. Reciprocal Rank Fusion is the preferred first implementation.

### 7. Reranking

The fused results are reranked before answer generation.

The reranker receives:

* user query
* candidate chunks

It returns the most relevant chunks for final context construction.

Preferred first implementation:

```txt
sentence-transformers CrossEncoder
```

### 8. Answer Generation

The answer generator uses an OpenAI-compatible chat model.

The prompt must enforce:

* answer only from retrieved context
* do not invent facts
* include sources
* say when there is not enough information
* keep answers concise

Expected answer format:

```txt
Answer:
...

Sources:
- filename.pdf, page 1, Section
- another_file.pdf, page 2, Section
```

### 9. Chainlit UI

Chainlit provides the chat interface.

The UI should show:

* user question
* generated answer
* source citations
* optionally retrieved chunks or debug information

A custom frontend is intentionally deferred until the RAG pipeline is working.

## Deployment

Docker Compose is the primary local deployment method.

Expected services:

```txt
app
qdrant
ollama
optional litellm
```

Helm is optional and should only be added after the local workflow is complete.

## Design Principles

* Local-first
* OpenAI-compatible model access
* Small, reviewable features
* Explicit retrieval pipeline
* Source-grounded answers
* Deterministic evaluation where possible
* Avoid unnecessary production infrastructure
