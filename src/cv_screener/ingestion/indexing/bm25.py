"""BM25 sparse vector construction using FastEmbed's Qdrant/bm25 model."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from fastembed import SparseTextEmbedding


@runtime_checkable
class BM25EncoderProtocol(Protocol):
    """Protocol for BM25 sparse encoders used by the indexer."""

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        """Encode a batch of texts into BM25 sparse vectors."""
        ...


class BM25Encoder:
    """Encode chunk text into BM25 sparse vectors using FastEmbed."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        """Initialize the FastEmbed BM25 sparse encoder."""
        self._model_name = model_name
        self._model = SparseTextEmbedding(model_name=model_name)

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        """Encode a batch of texts into BM25 sparse vectors.

        Returns a list of dicts mapping token index IDs to BM25 weights,
        one per input text, aligned with the input order.
        """
        results = list(self._model.embed(texts))
        return [
            {
                int(idx): float(val)
                for idx, val in zip(result.indices, result.values, strict=True)
            }
            for result in results
        ]
