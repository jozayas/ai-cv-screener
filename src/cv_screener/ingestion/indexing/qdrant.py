"""Qdrant-backed chunk indexing service."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from cv_screener.ingestion.indexing.bm25 import BM25Encoder, BM25EncoderProtocol
from cv_screener.ingestion.indexing.points import build_points
from cv_screener.ingestion.indexing.schema import QdrantIndexConfig

if TYPE_CHECKING:
    from cv_screener.ingestion.chunking.schema import Chunk
    from cv_screener.ingestion.indexing.protocols import (
        EmbeddingModel,
        QdrantIndexClientProtocol,
    )

_DISTANCE_BY_NAME = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
    "manhattan": models.Distance.MANHATTAN,
}


class QdrantChunkIndexer:
    """Index CV chunks into a Qdrant collection."""

    config: QdrantIndexConfig
    _client: QdrantIndexClientProtocol
    _embedding_model: EmbeddingModel
    _bm25_encoder: BM25EncoderProtocol | None

    def __init__(
        self,
        config: QdrantIndexConfig | None = None,
        *,
        client: QdrantIndexClientProtocol | None = None,
        embedding_model: EmbeddingModel | None = None,
        bm25_encoder: BM25EncoderProtocol | None = None,
    ) -> None:
        """Bind Qdrant and embedding dependencies to the indexing config."""
        self.config = config or QdrantIndexConfig()
        if client is not None:
            self._client = client
        else:
            real_client = QdrantClient(
                url=self.config.url,
                check_compatibility=self.config.check_compatibility,
            )
            self._client = cast("QdrantIndexClientProtocol", real_client)
        self._embedding_model = embedding_model or TextEmbedding(
            model_name=self.config.embedding_model_name,
        )
        if bm25_encoder is not None:
            self._bm25_encoder = bm25_encoder
        elif self.config.enable_bm25:
            self._bm25_encoder = BM25Encoder(model_name=self.config.bm25_model_name)
        else:
            self._bm25_encoder = None

    def ensure_collection(self, *, reset: bool = False) -> None:
        """Create the target collection if needed."""
        exists = self._client.collection_exists(self.config.collection_name)
        if exists and reset:
            self._client.delete_collection(self.config.collection_name)
            exists = False
        if exists:
            return

        sparse_vectors_config: dict[str, models.SparseVectorParams] | None = None
        if self.config.enable_bm25:
            vectors_config: models.VectorParams | dict[str, models.VectorParams] = {
                self.config.dense_vector_name: models.VectorParams(
                    size=self.config.vector_size,
                    distance=_DISTANCE_BY_NAME[self.config.distance],
                ),
            }
            sparse_vectors_config = {
                self.config.sparse_vector_name: models.SparseVectorParams(
                    index=models.SparseIndexParams(on_disk=False),
                    modifier=models.Modifier.IDF,
                ),
            }
        else:
            vectors_config = models.VectorParams(
                size=self.config.vector_size,
                distance=_DISTANCE_BY_NAME[self.config.distance],
            )

        self._client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )

    def index_chunks(self, chunks: list[Chunk], *, reset: bool = False) -> None:
        """Embed and upload chunks into Qdrant."""
        if not chunks:
            return

        self.ensure_collection(reset=reset)
        vectors = list(self._embedding_model.embed([chunk.text for chunk in chunks]))

        sparse_vectors: list[dict[int, float]] | None = None
        if self.config.enable_bm25 and self._bm25_encoder is not None:
            sparse_vectors = self._bm25_encoder.encode([chunk.text for chunk in chunks])

        points = build_points(
            chunks,
            vectors,
            config=self.config,
            sparse_vectors=sparse_vectors,
        )
        self._client.upload_points(
            collection_name=self.config.collection_name,
            points=points,
            batch_size=self.config.batch_size,
            parallel=self.config.parallel,
            max_retries=self.config.max_retries,
            wait=self.config.wait,
        )
