from __future__ import annotations

from typing import TYPE_CHECKING

from qdrant_client.http.models import (
    FusionQuery,
    Prefetch,
    QueryResponse,
    ScoredPoint,
)

import cv_screener.retrieval.hybrid as hybrid_module
from cv_screener.retrieval.hybrid import HybridRetriever
from cv_screener.retrieval.schema import HybridRetrievalConfig, RetrievedChunk

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pytest


class FakeEmbeddingModel:
    def __init__(self, vectors: list[list[float]] | None = None, **_: object) -> None:
        self._vectors: list[list[float]] = vectors or [[0.1, 0.2, 0.3]]

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
    ) -> list[list[float]]:
        del batch_size, parallel
        _ = documents
        return self._vectors


class FakeBM25Encoder:
    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        del model_name

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        return [{1: 1.0, 2: 0.5} for _ in texts]


class FakeQdrantClient:
    def __init__(self, **_: object) -> None:
        self.captured: dict[str, object] = {}
        self.response_points: list[ScoredPoint] = []

    def query_points(self, collection_name: str, **kwargs: object) -> QueryResponse:
        self.captured = {"collection_name": collection_name, **kwargs}
        return QueryResponse(points=self.response_points)


def _make_scored_point(
    *,
    score: float = 0.95,
    idx: int = 0,
    source_file: str = "candidate-a.pdf",
    text: str = "Built Python APIs.",
    section: str = "EXPERIENCE",
) -> ScoredPoint:
    return ScoredPoint(
        id=f"point-{idx}",
        version=0,
        score=score,
        payload={
            "candidate_name": "Candidate A",
            "source_file": source_file,
            "document_title": "Candidate A CV",
            "page": 1,
            "section": section,
            "text": text,
        },
        vector=None,
    )


def test_retrieve_uses_dense_and_sparse_prefetch() -> None:
    client = FakeQdrantClient()
    client.response_points = [_make_scored_point(idx=0)]
    retriever = HybridRetriever(
        config=HybridRetrievalConfig(
            semantic_top_k=5, bm25_top_k=7, fusion_top_k=3
        ),
        client=client,
        embedding_model=FakeEmbeddingModel(),
        bm25_encoder=FakeBM25Encoder(),
    )

    _ = retriever.retrieve("Python experience")

    captured = client.captured
    assert captured["collection_name"] == "cv_chunks"
    assert captured["limit"] == 3
    assert captured["with_payload"] is True
    assert captured["with_vectors"] is False

    prefetches = captured["prefetch"]
    assert isinstance(prefetches, list)
    assert len(prefetches) == 2

    dense_prefetch = prefetches[0]
    assert isinstance(dense_prefetch, Prefetch)
    assert dense_prefetch.using == "dense"
    assert dense_prefetch.limit == 5
    assert dense_prefetch.query == [0.1, 0.2, 0.3]

    sparse_prefetch = prefetches[1]
    assert isinstance(sparse_prefetch, Prefetch)
    assert sparse_prefetch.using == "bm25"
    assert sparse_prefetch.limit == 7

    fusion_query = captured["query"]
    assert isinstance(fusion_query, FusionQuery)


def test_retrieve_maps_payloads_to_retrieved_chunk() -> None:
    client = FakeQdrantClient()
    client.response_points = [
        _make_scored_point(score=0.95, idx=0),
        _make_scored_point(score=0.80, idx=1, text="Led engineering teams."),
    ]
    retriever = HybridRetriever(
        config=HybridRetrievalConfig(),
        client=client,
        embedding_model=FakeEmbeddingModel(),
        bm25_encoder=FakeBM25Encoder(),
    )

    chunks = retriever.retrieve("Python experience")

    assert len(chunks) == 2
    assert all(isinstance(c, RetrievedChunk) for c in chunks)

    c0 = chunks[0]
    assert c0.candidate_name == "Candidate A"
    assert c0.source_file == "candidate-a.pdf"
    assert c0.document_title == "Candidate A CV"
    assert c0.page == 1
    assert c0.section == "EXPERIENCE"
    assert c0.text == "Built Python APIs."
    assert c0.score == 0.95
    assert c0.rank == 1

    c1 = chunks[1]
    assert c1.score == 0.80
    assert c1.rank == 2
    assert c1.text == "Led engineering teams."


def test_retrieve_dense_only_when_bm25_disabled() -> None:
    client = FakeQdrantClient()
    client.response_points = [_make_scored_point(idx=0)]
    retriever = HybridRetriever(
        config=HybridRetrievalConfig(enable_bm25=False),
        client=client,
        embedding_model=FakeEmbeddingModel(),
        bm25_encoder=None,
    )

    _ = retriever.retrieve("Python experience")

    prefetches = client.captured["prefetch"]
    assert isinstance(prefetches, list)
    assert len(prefetches) == 1
    assert prefetches[0].using == "dense"


def test_retrieve_returns_empty_list_when_no_results() -> None:
    client = FakeQdrantClient()
    client.response_points = []
    retriever = HybridRetriever(
        config=HybridRetrievalConfig(),
        client=client,
        embedding_model=FakeEmbeddingModel(),
        bm25_encoder=FakeBM25Encoder(),
    )

    chunks = retriever.retrieve("nothing relevant")

    assert chunks == []


def test_retriever_uses_settings_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeClientWithCapture(FakeQdrantClient):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(**kwargs)
            captured.update(kwargs)

    monkeypatch.setattr(hybrid_module, "QdrantClient", FakeClientWithCapture)
    monkeypatch.setattr(hybrid_module, "TextEmbedding", FakeEmbeddingModel)
    monkeypatch.setattr(hybrid_module, "BM25Encoder", FakeBM25Encoder)

    _ = HybridRetriever()

    assert captured["url"] == "http://localhost:6333"
    assert captured["check_compatibility"] is False
