"""Qdrant point construction: stable IDs, payloads, and point structs."""

from collections.abc import Iterable
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import models
from qdrant_client.models import PointStruct

from cv_screener.ingestion.chunking.schema import Chunk
from cv_screener.ingestion.indexing.schema import QdrantIndexConfig


def document_id_for_chunk(chunk: Chunk) -> str:
    """Return a stable document identifier derived from chunk metadata."""
    identity = f"{chunk.source_file}\n{chunk.document_title}"
    return str(uuid5(NAMESPACE_URL, identity))


def point_id_for_chunk(chunk: Chunk) -> str:
    """Return a stable point identifier for a chunk payload."""
    identity = "\n".join(
        [
            document_id_for_chunk(chunk),
            str(chunk.page),
            str(chunk.chunk_index),
            chunk.section,
            chunk.text,
        ]
    )
    return str(uuid5(NAMESPACE_URL, identity))


def build_payload(chunk: Chunk) -> dict[str, str | int | list[str] | None]:
    """Build a Qdrant payload dict from a chunk."""
    return {
        "document_id": document_id_for_chunk(chunk),
        "candidate_name": chunk.candidate_name,
        "source_file": chunk.source_file,
        "document_title": chunk.document_title,
        "page": chunk.page,
        "chunk_index": chunk.chunk_index,
        "section": chunk.section,
        "detected_skills": chunk.detected_skills,
        "detected_companies": chunk.detected_companies,
        "detected_universities": chunk.detected_universities,
        "email_addresses": chunk.email_addresses,
        "phone_numbers": chunk.phone_numbers,
        "linkedin_urls": chunk.linkedin_urls,
        "github_urls": chunk.github_urls,
        "text": chunk.text,
    }


def _validate_embedding_size(
    vector_list: list[float], expected: int, chunk: Chunk
) -> None:
    """Raise ValueError if the embedding vector has the wrong size."""
    if len(vector_list) != expected:
        message = (
            f"Expected embedding size {expected}, "
            f"got {len(vector_list)} for chunk {chunk.source_file}#{chunk.chunk_index}"
        )
        raise ValueError(message)


def _build_point_vector(
    vector_list: list[float],
    *,
    config: QdrantIndexConfig,
    index: int,
    sparse_vectors: list[dict[int, float]],
) -> models.VectorStruct:
    """Construct the vector field for a Qdrant point with BM25 sparse vectors."""
    sparse = sparse_vectors[index]
    return {
        config.dense_vector_name: vector_list,
        config.sparse_vector_name: models.SparseVector(
            indices=list(sparse.keys()),
            values=list(sparse.values()),
        ),
    }


def build_points(
    chunks: list[Chunk],
    vectors: Iterable[Iterable[float]],
    *,
    config: QdrantIndexConfig,
    sparse_vectors: list[dict[int, float]] | None = None,
) -> list[PointStruct]:
    """Build Qdrant points for chunk/vector pairs.

    When BM25 is enabled (config.enable_bm25), each point carries both
    a named dense vector and a named sparse vector. Otherwise, points
    use a single unnamed dense vector.
    """
    has_sparse = config.enable_bm25 and sparse_vectors is not None
    points: list[PointStruct] = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors, strict=True)):
        vector_list = [float(value) for value in vector]
        _validate_embedding_size(vector_list, config.vector_size, chunk)

        if has_sparse and sparse_vectors is not None:
            point_vector = _build_point_vector(
                vector_list,
                config=config,
                index=i,
                sparse_vectors=sparse_vectors,
            )
        else:
            point_vector = vector_list

        points.append(
            models.PointStruct(
                id=point_id_for_chunk(chunk),
                vector=point_vector,
                payload=build_payload(chunk),
            )
        )
    return points
