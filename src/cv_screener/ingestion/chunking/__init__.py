"""Chunking internals for sectioning, semantic splitting, and metadata extraction."""

from cv_screener.ingestion.chunking.chunker import chunk_cv, chunk_cvs
from cv_screener.ingestion.chunking.metadata import (
    ChunkExtractionResult,
    MetadataExtractor,
)
from cv_screener.ingestion.chunking.schema import Chunk, ChunkConfig
from cv_screener.ingestion.chunking.sectioning import (
    MARKDOWN_SPLITTER,
    SectionClassifier,
    resolve_section,
)
from cv_screener.ingestion.chunking.semantic import SectionChunkRequest, SemanticChunker

__all__ = [
    "MARKDOWN_SPLITTER",
    "Chunk",
    "ChunkConfig",
    "ChunkExtractionResult",
    "MetadataExtractor",
    "SectionChunkRequest",
    "SectionClassifier",
    "SemanticChunker",
    "chunk_cv",
    "chunk_cvs",
    "resolve_section",
]
