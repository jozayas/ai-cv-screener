# Evaluation

## Goal

Evaluation should show that the system retrieves the right CVs and produces grounded answers with sources.

Use the generated YAML CV profiles as ground truth.

## Evaluation Types

The project should include two kinds of evaluation:

1. Retrieval evaluation
2. RAG answer evaluation

## Retrieval Evaluation

Retrieval evaluation checks whether the retrieval pipeline finds the right candidate chunks before answer generation.

The retrieval pipeline being evaluated is:

```txt
query
  → semantic search
  → BM25 search
  → fusion
  → reranking
  → final chunks
```

### Suggested Metrics

Use:

* Recall@5
* Precision@5
* MRR
* source hit rate

### Example Test Case

```yaml
id: kubernetes_experience
question: Who has experience with Kubernetes?
expected_candidates:
  - Ana Martín
  - David López
expected_terms:
  - Kubernetes
```

### What Counts as a Hit

A retrieved chunk is a hit if it belongs to one of the expected candidates.

For term-based questions, the chunk should also include or support the expected term where possible.

## RAG Answer Evaluation

RAG answer evaluation checks the final generated answer.

### Suggested Checks

For each test case, check:

* expected candidate appears in the answer
* answer includes source citations
* cited sources match retrieved context
* answer does not include unsupported candidates
* answer handles insufficient-information cases correctly

### Example Test Case

```yaml
id: upc_graduate
question: Which candidates studied at UPC?
expected_candidates:
  - Laura García
expected_terms:
  - UPC
requires_sources: true
```

### Insufficient Information Test

Include at least one question that should not be answerable.

Example:

```yaml
id: salary_expectations
question: What salary does Ana Martín expect?
expected_answer_type: insufficient_information
```

Expected behavior:

```txt
The available CVs do not contain enough information to answer that question.
```

## Deterministic Evaluation First

Prefer deterministic checks before LLM-as-judge.

Good deterministic checks:

* string match for expected candidate names
* string match for expected source filenames
* presence of `Sources:`
* no unexpected candidate names
* retrieval source hit rate

LLM-as-judge can be added later, but it should not replace deterministic evaluation.

## Evaluation Output

The evaluation command should print a concise summary:

```txt
Retrieval evaluation
Recall@5: 0.92
Precision@5: 0.71
MRR: 0.86
Source hit rate: 0.94

RAG evaluation
Expected candidate present: 9/10
Sources included: 10/10
Unsupported claims avoided: 8/10
Insufficient-information handled: 2/2
```

Optionally generate:

```txt
data/eval/report.md
```

## Recommended Evaluation Commands

```bash
python -m app.cli eval-retrieval
python -m app.cli eval-rag
```

## Minimum Evaluation Set

Start with 8–10 questions:

* 3 exact skill questions
* 2 education questions
* 2 language questions
* 1 ranking question
* 1 candidate summary question
* 1 insufficient-information question

Example questions:

```txt
Who has Python experience?
Which candidates have Kubernetes experience?
Which candidates speak German?
Who studied at UPC?
Who has experience with FastAPI and Docker?
Rank the best candidates for an AI backend role.
Summarize Ana Martín's profile.
What salary does Ana Martín expect?
```
