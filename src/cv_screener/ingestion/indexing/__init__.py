"""Helpers for indexing CV chunks into Qdrant."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cv_screener.ingestion.indexing.bm25 import BM25Encoder, BM25EncoderProtocol
    from cv_screener.ingestion.indexing.points import (
        build_payload,
        build_points,
        document_id_for_chunk,
        point_id_for_chunk,
    )
    from cv_screener.ingestion.indexing.qdrant import QdrantChunkIndexer
    from cv_screener.ingestion.indexing.schema import QdrantIndexConfig

__all__ = [
    "BM25Encoder",
    "BM25EncoderProtocol",
    "QdrantChunkIndexer",
    "QdrantIndexConfig",
    "build_payload",
    "build_points",
    "document_id_for_chunk",
    "point_id_for_chunk",
]


def __getattr__(name: str) -> object:
    """Load indexer exports on demand to avoid eager heavy imports."""
    if name in {"BM25Encoder", "BM25EncoderProtocol"}:
        bm25 = import_module("cv_screener.ingestion.indexing.bm25")
        return getattr(bm25, name)
    if name in {
        "build_payload",
        "build_points",
        "document_id_for_chunk",
        "point_id_for_chunk",
    }:
        points = import_module("cv_screener.ingestion.indexing.points")
        return getattr(points, name)
    if name == "QdrantChunkIndexer":
        return import_module("cv_screener.ingestion.indexing.qdrant").QdrantChunkIndexer
    if name == "QdrantIndexConfig":
        return import_module("cv_screener.ingestion.indexing.schema").QdrantIndexConfig
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
