from typing import TYPE_CHECKING, cast

import pytest

import cv_screener.ingestion.indexing.qdrant as qdrant_module
from cv_screener.ingestion.chunking.schema import Chunk
from cv_screener.ingestion.indexing.qdrant import QdrantChunkIndexer
from cv_screener.ingestion.indexing.schema import QdrantIndexConfig

if TYPE_CHECKING:
    from qdrant_client import models


class FakeEmbeddingModel:
    def __init__(self, vectors: list[list[float]]) -> None:
        self._vectors = vectors
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(texts)
        return self._vectors


class FakeQdrantClient:
    def __init__(self, *, exists: bool) -> None:
        self.exists = exists
        self.created: list[tuple[str, object]] = []
        self.deleted: list[str] = []
        self.uploads: list[tuple[str, list[object], int, int, int, bool]] = []

    def collection_exists(self, collection_name: str) -> bool:
        del collection_name
        return self.exists

    def create_collection(self, collection_name: str, vectors_config: object) -> None:
        self.created.append((collection_name, vectors_config))
        self.exists = True

    def delete_collection(self, collection_name: str) -> None:
        self.deleted.append(collection_name)
        self.exists = False

    def upload_points(
        self,
        collection_name: str,
        points: object,
        **kwargs: object,
    ) -> object:
        self.uploads.append(
            (
                collection_name,
                cast("list[object]", points),
                cast("int", kwargs["batch_size"]),
                cast("int", kwargs["parallel"]),
                cast("int", kwargs["max_retries"]),
                cast("bool", kwargs["wait"]),
            )
        )
        return None


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
    indexer = QdrantChunkIndexer(
        config=QdrantIndexConfig(vector_size=3),
        client=FakeQdrantClient(exists=False),
        embedding_model=FakeEmbeddingModel([[0.1, 0.2, 0.3]]),
    )

    payload = indexer.build_payload(chunk)

    assert payload["candidate_name"] == "Marta Alvarez"
    assert payload["source_file"] == "marta-alvarez.pdf"
    assert payload["document_title"] == "Marta Alvarez CV"
    assert payload["page"] == 1
    assert payload["chunk_index"] == 2
    assert payload["section"] == "EXPERIENCE"
    assert payload["detected_skills"] == ["Python", "FastAPI"]
    assert payload["text"] == "Built Python APIs at NovaStack."
    assert payload["document_id"] == indexer.document_id_for_chunk(chunk)


def test_point_id_is_deterministic_for_same_chunk() -> None:
    chunk = _make_chunk()
    indexer = QdrantChunkIndexer(
        config=QdrantIndexConfig(vector_size=3),
        client=FakeQdrantClient(exists=False),
        embedding_model=FakeEmbeddingModel([[0.1, 0.2, 0.3]]),
    )

    point_id = indexer.point_id_for_chunk(chunk)

    assert point_id == indexer.point_id_for_chunk(chunk)
    assert point_id != indexer.point_id_for_chunk(_make_chunk(chunk_index=3))


def test_ensure_collection_creates_missing_collection() -> None:
    client = FakeQdrantClient(exists=False)
    indexer = QdrantChunkIndexer(
        config=QdrantIndexConfig(vector_size=3, distance="dot"),
        client=client,
        embedding_model=FakeEmbeddingModel([[0.1, 0.2, 0.3]]),
    )

    indexer.ensure_collection()

    assert client.created
    collection_name, vectors_config = client.created[0]
    assert collection_name == "cv_chunks"
    vector_params = cast("models.VectorParams", vectors_config)
    assert vector_params.size == 3
    assert vector_params.distance.name == "DOT"


def test_ensure_collection_recreates_on_reset() -> None:
    client = FakeQdrantClient(exists=True)
    indexer = QdrantChunkIndexer(
        config=QdrantIndexConfig(vector_size=3),
        client=client,
        embedding_model=FakeEmbeddingModel([[0.1, 0.2, 0.3]]),
    )

    indexer.ensure_collection(reset=True)

    assert client.deleted == ["cv_chunks"]
    assert len(client.created) == 1


def test_index_chunks_embeds_text_and_uploads_points() -> None:
    chunk = _make_chunk()
    client = FakeQdrantClient(exists=False)
    embedding_model = FakeEmbeddingModel([[0.1, 0.2, 0.3]])
    indexer = QdrantChunkIndexer(
        config=QdrantIndexConfig(
            vector_size=3, batch_size=16, parallel=2, max_retries=5, wait=True
        ),
        client=client,
        embedding_model=embedding_model,
    )

    indexer.index_chunks([chunk])

    assert embedding_model.calls == [[chunk.text]]
    assert len(client.uploads) == 1
    collection_name, points, batch_size, parallel, max_retries, wait = client.uploads[0]
    assert collection_name == "cv_chunks"
    assert batch_size == 16
    assert parallel == 2
    assert max_retries == 5
    assert wait
    point = cast("models.PointStruct", points[0])
    assert point.id == indexer.point_id_for_chunk(chunk)
    assert point.vector == [0.1, 0.2, 0.3]
    assert point.payload is not None
    assert point.payload["source_file"] == "marta-alvarez.pdf"


def test_indexer_uses_qdrant_settings_for_default_connection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeQdrantClient:
        def __init__(self, **kwargs: object) -> None:
            captured.update(kwargs)

    class FakeTextEmbedding:
        def __init__(self, model_name: str) -> None:
            del model_name

    monkeypatch.setattr(qdrant_module, "QdrantClient", FakeQdrantClient)
    monkeypatch.setattr(qdrant_module, "TextEmbedding", FakeTextEmbedding)

    QdrantChunkIndexer()

    assert captured["url"] == "http://localhost:6333"
    assert captured["check_compatibility"] is False
