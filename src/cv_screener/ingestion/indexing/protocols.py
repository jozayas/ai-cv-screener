"""Typing protocols for indexer dependencies."""

from collections.abc import Iterable
from typing import Protocol


class EmbeddingModel(Protocol):
    """Protocol for embedding models used by the indexer."""

    def embed(self, texts: list[str]) -> Iterable[Iterable[float]]:
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
        vectors_config: object = None,
        **kwargs: object,
    ) -> object:
        """Create a collection with vector configuration."""

    def delete_collection(self, collection_name: str, **kwargs: object) -> object:
        """Delete an existing collection."""

    def upload_points(  # noqa: PLR0913
        self,
        collection_name: str,
        points: object,
        *,
        batch_size: int = 64,
        parallel: int = 1,
        max_retries: int = 3,
        wait: bool = True,
    ) -> object:
        """Upload points into a collection."""
