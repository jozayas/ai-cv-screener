from __future__ import annotations

from typing import TYPE_CHECKING, cast

from qdrant_client.http.models import SparseVectorParams, VectorParams

import cv_screener.ingestion.indexing.qdrant as qdrant_module
from cv_screener.ingestion.chunking.schema import Chunk
from cv_screener.ingestion.indexing.points import (
    build_payload,
    document_id_for_chunk,
    point_id_for_chunk,
)
from cv_screener.ingestion.indexing.qdrant import QdrantChunkIndexer
from cv_screener.ingestion.indexing.schema import QdrantIndexConfig

if TYPE_CHECKING:
    from collections.abc import Iterable

    import pytest
    from qdrant_client.http.models import PointStruct


class FakeEmbeddingModel:
    def __init__(
        self, vectors: list[list[float]] | None = None, **_kwargs: object
    ) -> None:
        self._vectors: list[list[float]] = vectors or [[0.1, 0.2, 0.3]]
        self.calls: list[list[str]] = []

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
    ) -> list[list[float]]:
        del batch_size, parallel
        if isinstance(documents, str):
            texts: list[str] = [documents]
        else:
            texts = list(documents)
        self.calls.append(texts)
        return self._vectors


class FakeBM25Encoder:
    """Fake BM25 encoder compatible with BM25EncoderProtocol."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        del model_name

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        return [{1: 1.0, 2: 0.5} for _ in texts]


class FakeQdrantClient:
    """Fake Qdrant client that captures calls for test assertions."""

    def __init__(self, *, exists: bool = False, **_kwargs: object) -> None:
        self.exists: bool = exists
        self.created: list[tuple[str, object, object]] = []
        self.deleted: list[str] = []
        self.uploads: list[tuple[str, list[PointStruct], int, int, int, bool]] = []

    def collection_exists(self, collection_name: str) -> bool:
        del collection_name
        return self.exists

    def create_collection(
        self,
        collection_name: str,
        vectors_config: VectorParams | dict[str, VectorParams] | None = None,
        sparse_vectors_config: dict[str, SparseVectorParams] | None = None,
        **_kwargs: object,
    ) -> bool:
        self.created.append((collection_name, vectors_config, sparse_vectors_config))
        self.exists = True
        return True

    def delete_collection(self, collection_name: str, **_kwargs: object) -> bool:
        self.deleted.append(collection_name)
        self.exists = False
        return True

    def upload_points(
        self,
        collection_name: str,
        points: Iterable[PointStruct],
        **kwargs: object,
    ) -> None:
        batch_size = cast("int", kwargs.get("batch_size", 64))
        parallel = cast("int", kwargs.get("parallel", 1))
        max_retries = cast("int", kwargs.get("max_retries", 3))
        wait = cast("bool", kwargs.get("wait", True))
        self.uploads.append(
            (collection_name, list(points), batch_size, parallel, max_retries, wait)
        )


def _make_chunk(
    **overrides: str | int | list[str] | None,
) -> Chunk:
    base: dict[str, str | int | list[str] | None] = {
        "candidate_name": "Marta Alvarez",
        "source_file": "marta-alvarez.pdf",
        "document_title": "Marta Alvarez CV",
        "page": 1,
        "chunk_index": 2,
        "section": "EXPERIENCE",
        "detected_skills": ["Python", "FastAPI"],
        "detected_companies": ["NovaStack"],
        "detected_universities": [],
        "email_addresses": ["marta@example.com"],
        "phone_numbers": ["+34600123123"],
        "linkedin_urls": ["https://linkedin.com/in/marta"],
        "github_urls": ["https://github.com/marta"],
        "text": "Built Python APIs at NovaStack.",
    }
    base.update(overrides)
    return Chunk.model_validate(base)


def test_payload_contains_chunk_metadata_and_index_identity() -> None:
    chunk = _make_chunk()

    payload = build_payload(chunk)

    assert payload["candidate_name"] == "Marta Alvarez"
    assert payload["source_file"] == "marta-alvarez.pdf"
    assert payload["document_title"] == "Marta Alvarez CV"
    assert payload["page"] == 1
    assert payload["chunk_index"] == 2
    assert payload["section"] == "EXPERIENCE"
    assert payload["detected_skills"] == ["Python", "FastAPI"]
    assert payload["text"] == "Built Python APIs at NovaStack."
    assert payload["document_id"] == document_id_for_chunk(chunk)


def test_point_id_is_deterministic_for_same_chunk() -> None:
    chunk = _make_chunk()

    pid = point_id_for_chunk(chunk)

    assert pid == point_id_for_chunk(chunk)
    assert pid != point_id_for_chunk(_make_chunk(chunk_index=3))


def test_ensure_collection_with_bm25_uses_named_dense_and_sparse_config() -> None:
    client = FakeQdrantClient(exists=False)
    indexer = QdrantChunkIndexer(
        config=QdrantIndexConfig(vector_size=3, distance="dot", enable_bm25=True),
        client=client,
        embedding_model=FakeEmbeddingModel([[0.1, 0.2, 0.3]]),
        bm25_encoder=FakeBM25Encoder(),
    )

    indexer.ensure_collection()

    assert client.created
    collection_name, vectors_config, sparse_config = client.created[0]
    assert collection_name == "cv_chunks"
    assert isinstance(vectors_config, dict)
    assert "dense" in vectors_config
    assert isinstance(vectors_config["dense"], VectorParams)
    assert vectors_config["dense"].size == 3
    assert vectors_config["dense"].distance.name == "DOT"
    assert isinstance(sparse_config, dict)
    assert "bm25" in sparse_config
    assert isinstance(sparse_config["bm25"], SparseVectorParams)


def test_ensure_collection_without_bm25_uses_unnamed_dense_config() -> None:
    client = FakeQdrantClient(exists=False)
    indexer = QdrantChunkIndexer(
        config=QdrantIndexConfig(vector_size=3, distance="dot", enable_bm25=False),
        client=client,
        embedding_model=FakeEmbeddingModel([[0.1, 0.2, 0.3]]),
    )

    indexer.ensure_collection()

    assert client.created
    collection_name, vectors_config, sparse_config = client.created[0]
    assert collection_name == "cv_chunks"
    assert isinstance(vectors_config, VectorParams)
    assert vectors_config.size == 3
    assert vectors_config.distance.name == "DOT"
    assert sparse_config is None


def test_ensure_collection_recreates_on_reset() -> None:
    client = FakeQdrantClient(exists=True)
    indexer = QdrantChunkIndexer(
        config=QdrantIndexConfig(vector_size=3),
        client=client,
        embedding_model=FakeEmbeddingModel([[0.1, 0.2, 0.3]]),
        bm25_encoder=FakeBM25Encoder(),
    )

    indexer.ensure_collection(reset=True)

    assert client.deleted == ["cv_chunks"]
    assert len(client.created) == 1


def test_index_chunks_embeds_text_and_uploads_points_with_bm25() -> None:
    chunk = _make_chunk()
    client = FakeQdrantClient(exists=False)
    embedding_model = FakeEmbeddingModel([[0.1, 0.2, 0.3]])
    bm25_encoder = FakeBM25Encoder()
    indexer = QdrantChunkIndexer(
        config=QdrantIndexConfig(
            vector_size=3, batch_size=16, parallel=2, max_retries=5, wait=True
        ),
        client=client,
        embedding_model=embedding_model,
        bm25_encoder=bm25_encoder,
    )

    indexer.index_chunks([chunk])

    assert embedding_model.calls == [[chunk.text]]
    assert len(client.uploads) == 1
    _, points, batch_size, parallel, max_retries, wait = client.uploads[0]
    assert batch_size == 16
    assert parallel == 2
    assert max_retries == 5
    assert wait
    point = next(iter(points))
    assert point.id == point_id_for_chunk(chunk)
    assert isinstance(point.vector, dict)
    assert "dense" in point.vector
    assert point.vector["dense"] == [0.1, 0.2, 0.3]
    assert "bm25" in point.vector
    assert point.payload is not None
    assert point.payload["source_file"] == "marta-alvarez.pdf"


def test_index_chunks_without_bm25_uses_unnamed_vectors() -> None:
    chunk = _make_chunk()
    client = FakeQdrantClient(exists=False)
    embedding_model = FakeEmbeddingModel([[0.1, 0.2, 0.3]])
    indexer = QdrantChunkIndexer(
        config=QdrantIndexConfig(vector_size=3, enable_bm25=False),
        client=client,
        embedding_model=embedding_model,
    )

    indexer.index_chunks([chunk])

    assert len(client.uploads) == 1
    _, points, *_ = client.uploads[0]
    point = next(iter(points))
    assert point.vector == [0.1, 0.2, 0.3]


def test_indexer_uses_qdrant_settings_for_default_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeQdrantClientWithCapture(FakeQdrantClient):
        def __init__(self, **kwargs: object) -> None:
            super().__init__(exists=False, **kwargs)
            captured.update(kwargs)

    monkeypatch.setattr(qdrant_module, "QdrantClient", FakeQdrantClientWithCapture)
    monkeypatch.setattr(qdrant_module, "TextEmbedding", FakeEmbeddingModel)
    monkeypatch.setattr(qdrant_module, "BM25Encoder", FakeBM25Encoder)

    _ = QdrantChunkIndexer()

    assert captured["url"] == "http://localhost:6333"
    assert captured["check_compatibility"] is False
