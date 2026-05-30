"""Ingestion package: PDF parsing, chunking, and indexing."""

from cv_screener.ingestion.chunking import chunk_cv, chunk_cvs
from cv_screener.ingestion.indexing import QdrantChunkIndexer
from cv_screener.ingestion.ingest import CVIngestionService, IngestionSummary
from cv_screener.ingestion.parser import parse_directory, parse_pdf

__all__ = [
    "CVIngestionService",
    "IngestionSummary",
    "QdrantChunkIndexer",
    "chunk_cv",
    "chunk_cvs",
    "parse_directory",
    "parse_pdf",
]
