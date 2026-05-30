"""Qdrant-backed chunk indexing service."""

from typing import TYPE_CHECKING, cast

from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models

from cv_screener.ingestion.chunking.schema import Chunk
from cv_screener.ingestion.indexing.ids import (
    document_id_for_chunk,
    point_id_for_chunk,
)
from cv_screener.ingestion.indexing.payloads import build_payload
from cv_screener.ingestion.indexing.points import build_points
from cv_screener.ingestion.indexing.schema import QdrantIndexConfig

if TYPE_CHECKING:
    from cv_screener.ingestion.indexing.protocols import (
        EmbeddingModel,
        QdrantClientProtocol,
    )

_DISTANCE_BY_NAME = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
    "manhattan": models.Distance.MANHATTAN,
}


class QdrantChunkIndexer:
    """Index CV chunks into a Qdrant collection."""

    def __init__(
        self,
        config: QdrantIndexConfig | None = None,
        *,
        client: object | None = None,
        embedding_model: object | None = None,
    ) -> None:
        """Bind Qdrant and embedding dependencies to the indexing config."""
        self.config = config or QdrantIndexConfig()
        self._client = cast(
            "QdrantClientProtocol",
            client
            or QdrantClient(
                url=self.config.url,
                check_compatibility=self.config.check_compatibility,
            ),
        )
        self._embedding_model = cast(
            "EmbeddingModel",
            embedding_model
            or TextEmbedding(model_name=self.config.embedding_model_name),
        )

    def document_id_for_chunk(self, chunk: Chunk) -> str:
        """Return a stable document identifier derived at indexing time."""
        return document_id_for_chunk(chunk)

    def point_id_for_chunk(self, chunk: Chunk) -> str:
        """Return a stable point identifier for a chunk payload."""
        return point_id_for_chunk(chunk)

    def build_payload(self, chunk: Chunk) -> dict[str, str | int | list[str] | None]:
        """Build a Qdrant payload from a chunk."""
        return build_payload(chunk)

    def ensure_collection(self, *, reset: bool = False) -> None:
        """Create the target collection if needed."""
        exists = self._client.collection_exists(self.config.collection_name)
        if exists and reset:
            self._client.delete_collection(self.config.collection_name)
            exists = False
        if exists:
            return
        self._client.create_collection(
            collection_name=self.config.collection_name,
            vectors_config=models.VectorParams(
                size=self.config.vector_size,
                distance=_DISTANCE_BY_NAME[self.config.distance],
            ),
        )

    def index_chunks(self, chunks: list[Chunk], *, reset: bool = False) -> None:
        """Embed and upload chunks into Qdrant."""
        if not chunks:
            return

        self.ensure_collection(reset=reset)
        vectors = list(self._embedding_model.embed([chunk.text for chunk in chunks]))
        points = build_points(chunks, vectors, config=self.config)
        self._client.upload_points(
            collection_name=self.config.collection_name,
            points=points,
            batch_size=self.config.batch_size,
            parallel=self.config.parallel,
            max_retries=self.config.max_retries,
            wait=self.config.wait,
        )
