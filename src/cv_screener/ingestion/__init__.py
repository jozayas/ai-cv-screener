"""Ingestion package: PDF parsing, chunking, and indexing."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cv_screener.ingestion.chunking import chunk_cv, chunk_cvs
    from cv_screener.ingestion.indexing import QdrantChunkIndexer
    from cv_screener.ingestion.ingest import CVIngestionService
    from cv_screener.ingestion.parser import parse_directory, parse_pdf
    from cv_screener.ingestion.schema import IngestionSummary

__all__ = [
    "CVIngestionService",
    "IngestionSummary",
    "QdrantChunkIndexer",
    "chunk_cv",
    "chunk_cvs",
    "parse_directory",
    "parse_pdf",
]


def __getattr__(name: str) -> object:
    """Load ingestion exports on demand to keep package import cheap."""
    if name in {"chunk_cv", "chunk_cvs"}:
        chunking = import_module("cv_screener.ingestion.chunking")
        return getattr(chunking, name)
    if name == "QdrantChunkIndexer":
        indexing = import_module("cv_screener.ingestion.indexing")
        return indexing.QdrantChunkIndexer
    if name == "CVIngestionService":
        return import_module("cv_screener.ingestion.ingest").CVIngestionService
    if name == "IngestionSummary":
        return import_module("cv_screener.ingestion.schema").IngestionSummary
    if name in {"parse_directory", "parse_pdf"}:
        parser = import_module("cv_screener.ingestion.parser")
        return getattr(parser, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
