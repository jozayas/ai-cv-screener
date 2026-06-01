from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import pytest
from qdrant_client import QdrantClient

from cv_screener.ingestion.chunking.schema import Chunk
from cv_screener.ingestion.indexing.points import (
    document_id_for_chunk,
    point_id_for_chunk,
)
from cv_screener.ingestion.indexing.qdrant import QdrantChunkIndexer
from cv_screener.ingestion.indexing.schema import QdrantIndexConfig

if TYPE_CHECKING:
    from collections.abc import Iterable


class FakeEmbeddingModel:
    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
    ) -> list[list[float]]:
        del batch_size, parallel
        texts = [documents] if isinstance(documents, str) else list(documents)
        assert len(texts) == 1
        return [[0.1, 0.2, 0.3]]


@pytest.mark.integration
def test_qdrant_chunk_indexer_indexes_into_live_qdrant() -> None:
    chunk = Chunk(
        candidate_name="Marta Alvarez",
        source_file="marta-alvarez.pdf",
        document_title="Marta Alvarez CV",
        page=1,
        chunk_index=0,
        section="SUMMARY",
        detected_skills=["Python"],
        detected_companies=[],
        detected_universities=[],
        email_addresses=["marta@example.com"],
        phone_numbers=[],
        linkedin_urls=[],
        github_urls=[],
        text="Python backend engineer.",
    )
    collection_name = "cv_chunks_integration"
    client = QdrantClient(url="http://localhost:6333")
    indexer = QdrantChunkIndexer(
        config=QdrantIndexConfig(collection_name=collection_name, vector_size=3),
        client=client,
        embedding_model=FakeEmbeddingModel(),
    )

    indexer.index_chunks([chunk], reset=True)

    stored_points = client.retrieve(
        collection_name=collection_name,
        ids=[UUID(point_id_for_chunk(chunk))],
        with_payload=True,
        with_vectors=False,
    )

    assert len(stored_points) == 1
    assert stored_points[0].payload is not None
    assert stored_points[0].payload["source_file"] == "marta-alvarez.pdf"
    assert stored_points[0].payload["document_id"] == document_id_for_chunk(chunk)
