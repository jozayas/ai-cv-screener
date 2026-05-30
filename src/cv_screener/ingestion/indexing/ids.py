"""Stable identifiers for indexed documents and chunks."""

from uuid import NAMESPACE_URL, uuid5

from cv_screener.ingestion.chunking.schema import Chunk


def document_id_for_chunk(chunk: Chunk) -> str:
    """Return a stable document identifier derived at indexing time."""
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
