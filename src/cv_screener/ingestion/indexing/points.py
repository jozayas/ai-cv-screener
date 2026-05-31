"""Point construction helpers for Qdrant indexing."""

from collections.abc import Iterable
from typing import TYPE_CHECKING

from qdrant_client import models

from cv_screener.ingestion.chunking.schema import Chunk
from cv_screener.ingestion.indexing.ids import point_id_for_chunk
from cv_screener.ingestion.indexing.payloads import build_payload
from cv_screener.ingestion.indexing.schema import QdrantIndexConfig

if TYPE_CHECKING:
    from qdrant_client.models import PointStruct


def build_points(
    chunks: list[Chunk],
    vectors: Iterable[Iterable[float]],
    *,
    config: QdrantIndexConfig,
) -> list["PointStruct"]:
    """Build Qdrant points for chunk/vector pairs."""
    points: list[PointStruct] = []
    for chunk, vector in zip(chunks, vectors, strict=True):
        vector_list = [float(value) for value in vector]
        if len(vector_list) != config.vector_size:
            message = (
                f"Expected embedding size {config.vector_size}, "
                f"got {len(vector_list)} for chunk {chunk.source_file}#{chunk.chunk_index}"
            )
            raise ValueError(message)
        points.append(
            models.PointStruct(
                id=point_id_for_chunk(chunk),
                vector=vector_list,
                payload=build_payload(chunk),
            )
        )
    return points
