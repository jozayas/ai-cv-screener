# AI-Powered CV Screener

Local-first prototype for screening fake CVs with a hybrid RAG pipeline.

The application generates realistic fake CVs as PDFs, parses and ingests them, and exposes a chat interface where users can ask questions about the candidate pool. Answers are grounded in the parsed CV contents and include source citations.

## Features

* Generate fake candidate profiles as YAML
* Validate CV YAML files with Pydantic
* Render CVs to PDF
* Parse PDFs with PyMuPDF
* Chunk CV content using section-aware and semantic chunking
* Index content in Qdrant for semantic search
* Index content with BM25 for keyword search
* Use hybrid retrieval with fusion
* Rerank retrieved chunks before answer generation
* Answer questions using an OpenAI-compatible model client
* Run local models through Ollama
* Use Chainlit as the chat interface
* Evaluate retrieval and answer quality
* Run locally with Docker Compose

## Stack

* Python 3.12
* uv
* Typer
* Pydantic
* YAML
* Jinja2
* WeasyPrint
* PyMuPDF
* LangChain
* Qdrant
* BM25
* sentence-transformers reranker
* OpenAI-compatible model API
* Ollama
* Chainlit
* Docker Compose

## Project Structure

```txt
.
├── AGENTS.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .python-version
├── .env.example
├── docker-compose.yml
├── Dockerfile
│
├── docs/
│   ├── architecture.md
│   └── evaluation.md
│
├── src/
│   └── cv_screener/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── chainlit_app.py
│       │
│       ├── cv_generation/
│       │   ├── __init__.py
│       │   ├── schema.py
│       │   ├── generator.py
│       │   ├── renderer.py
│       │   └── templates/
│       │       └── cv.html.j2
│       │
│       ├── ingestion/
│       │   ├── __init__.py
│       │   ├── parser.py
│       │   ├── chunker.py
│       │   └── indexer.py
│       │
│       ├── retrieval/
│       │   ├── __init__.py
│       │   ├── semantic.py
│       │   ├── bm25.py
│       │   ├── fusion.py
│       │   ├── reranker.py
│       │   └── hybrid.py
│       │
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── prompts.py
│       │   └── chain.py
│       │
│       └── evaluation/
│           ├── __init__.py
│           ├── datasets.py
│           ├── retrieval_eval.py
│           └── rag_eval.py
│
├── tests/
│   ├── test_schema.py
│   ├── test_chunker.py
│   └── test_fusion.py
│
└── data/
    ├── cvs_contents/
    ├── cvs_pdf/
    ├── parsed/
    ├── indexes/
    └── eval/
```

## Configuration

Create a `.env` file from `.env.example`.

```env
OPENAI_BASE_URL=http://localhost:11434/v1
OPENAI_API_KEY=ollama

GENERATION_MODEL=llama3.1:8b
CHAT_MODEL=llama3.1:8b
EMBEDDING_MODEL=nomic-embed-text

QDRANT_URL=http://localhost:6333
QDRANT_COLLECTION=cv_chunks

SEMANTIC_TOP_K=12
BM25_TOP_K=12
FUSION_TOP_K=10
RERANK_TOP_K=5
```

## Local Setup

Pin Python 3.12:

```bash
uv python pin 3.12
```

Install dependencies:

```bash
uv sync
```

Start local infrastructure:

```bash
docker compose up -d qdrant ollama
```

Pull local models if using Ollama:

```bash
ollama pull llama3.1:8b
ollama pull nomic-embed-text
```

## CLI Usage

The project exposes a Typer CLI through the `cv-screener` command.

Generate fake CV YAML profiles:

```bash
uv run cv-screener generate-cvs --count 30 --output-dir data/cvs_contents
```

Validate generated YAML files:

```bash
uv run cv-screener validate-cvs --input-dir data/cvs_contents
```

Render CV PDFs:

```bash
uv run cv-screener render-pdfs
```

Ingest PDFs into the RAG indexes:

```bash
uv run cv-screener ingest --reset
```

Ask a question from the CLI:

```bash
uv run cv-screener query "Who has experience with Python?"
```

Run the Chainlit app:

```bash
uv run cv-screener serve
```

Or directly:

```bash
uv run chainlit run src/cv_screener/chainlit_app.py
```

Run retrieval evaluation:

```bash
uv run cv-screener eval-retrieval
```

Run RAG evaluation:

```bash
uv run cv-screener eval-rag
```

## Example Questions

```txt
Who has experience with Python?
Which candidates speak German?
Who studied at UPC?
Which candidates have Kubernetes and FastAPI experience?
Rank the best candidates for an AI backend role.
Summarize the profile of Ana Martín.
Which candidates would be suitable for a data engineering role?
```

## RAG Behavior

The system should answer only from retrieved CV content.

Every answer should include sources, for example:

```txt
Answer:
Ana Martín and David López have Python experience. Ana used Python in backend AI services, while David used it for data pipelines.

Sources:
- ana_martin.pdf, page 1, Experience
- david_lopez.pdf, page 1, Skills
```

If the answer is not supported by the CVs, the system should say that the available CVs do not contain enough information.

## Development Workflow

Build the project incrementally.

Recommended order:

1. Pydantic CV schema
2. YAML validation
3. LLM-based YAML generation
4. PDF rendering
5. PDF parsing
6. Chunking
7. Qdrant indexing
8. BM25 indexing
9. Hybrid retrieval
10. Reranking
11. RAG answering
12. Chainlit UI
13. Evaluation
14. Docker Compose
15. Demo instructions

Start with 3 CVs. Scale to 25–30 after the full pipeline works.

## Demo Flow

Suggested demo structure:

1. Show generated YAML profiles.
2. Show generated PDF CVs.
3. Run ingestion.
4. Open Chainlit.
5. Ask exact-match questions.
6. Ask semantic ranking questions.
7. Show source citations.
8. Show evaluation output.
9. Briefly explain the hybrid retrieval and reranking pipeline.
