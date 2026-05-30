"""Chunking internals for sectioning, semantic splitting, and metadata extraction."""

from cv_screener.ingestion.chunking.metadata import (
    ChunkExtractionResult,
    MetadataExtractor,
)
from cv_screener.ingestion.chunking.sectioning import (
    MARKDOWN_SPLITTER,
    SectionClassifier,
    resolve_section,
)
from cv_screener.ingestion.chunking.semantic import SectionChunkRequest, SemanticChunker

__all__ = [
    "MARKDOWN_SPLITTER",
    "ChunkExtractionResult",
    "MetadataExtractor",
    "SectionChunkRequest",
    "SectionClassifier",
    "SemanticChunker",
    "resolve_section",
]
