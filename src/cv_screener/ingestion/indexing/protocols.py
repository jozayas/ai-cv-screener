"""Typing protocols for indexer dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

    from qdrant_client.http.models import PointStruct, VectorParams


class EmbeddingModel(Protocol):
    """Protocol for embedding models used by the indexer."""

    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
    ) -> Iterable[Iterable[float]]:
        """Embed a batch of texts."""
        ...


class QdrantClientProtocol(Protocol):
    """Protocol for the subset of Qdrant client features the indexer needs."""

    def collection_exists(self, collection_name: str) -> bool:
        """Return whether the collection already exists."""
        ...

    def create_collection(
        self,
        collection_name: str,
        vectors_config: VectorParams | dict[str, VectorParams] | None = None,
        **kwargs: Any,  # noqa: ANN401
    ) -> bool:
        """Create a collection with vector configuration."""
        ...

    def delete_collection(self, collection_name: str, **kwargs: Any) -> bool:  # noqa: ANN401
        """Delete an existing collection."""
        ...

    def upload_points(  # noqa: PLR0913
        self,
        collection_name: str,
        points: Iterable[PointStruct],
        *,
        batch_size: int = 64,
        parallel: int = 1,
        max_retries: int = 3,
        wait: bool = True,
    ) -> None:
        """Upload points into a collection."""
        ...
