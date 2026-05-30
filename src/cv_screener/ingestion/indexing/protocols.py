"""Typing protocols for indexer dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from collections.abc import Iterable

    from qdrant_client.http.models import (
        PointStruct,
        QueryResponse,
    )


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


class QdrantIndexClientProtocol(Protocol):
    """Protocol for the subset of Qdrant client features the indexer needs."""

    def collection_exists(self, collection_name: str) -> bool:
        """Return whether the collection already exists."""
        ...

    def create_collection(
        self,
        collection_name: str,
        **kwargs: object,
    ) -> bool:
        """Create a collection with vector configuration."""
        ...

    def delete_collection(self, collection_name: str, **kwargs: object) -> bool:
        """Delete an existing collection."""
        ...

    def upload_points(
        self,
        collection_name: str,
        points: Iterable[PointStruct],
        **kwargs: object,
    ) -> None:
        """Upload points into a collection."""
        ...

class QdrantQueryClientProtocol(Protocol):
    """Protocol for the subset of Qdrant client features the retriever needs."""

    def query_points(
        self,
        collection_name: str,
        **kwargs: object,
    ) -> QueryResponse:
        """Query points from a collection."""
        ...
