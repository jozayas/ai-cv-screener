"""Orchestration for parsing, chunking, and indexing rendered CV PDFs."""
from pathlib import Path
from typing import Protocol

from cv_screener.ingestion.chunking import chunk_cvs
from cv_screener.ingestion.chunking.schema import Chunk
from cv_screener.ingestion.indexing import QdrantChunkIndexer
from cv_screener.ingestion.parser import parse_directory
from cv_screener.ingestion.parsing.schema import ParsedCV
from cv_screener.ingestion.schema import IngestionSummary


class CVDirectoryParser(Protocol):
    """Callable interface for parsing a directory of rendered CV PDFs."""

    def __call__(self, directory: Path) -> list[ParsedCV]:
        """Parse a directory of PDFs into structured CV documents."""
        ...


class CVChunker(Protocol):
    """Callable interface for chunking parsed CV documents."""

    def __call__(self, cvs: list[ParsedCV]) -> list[Chunk]:
        """Convert parsed CVs into retrieval-ready chunks."""
        ...


class ChunkIndexer(Protocol):
    """Interface for indexing retrieval-ready chunks."""

    def index_chunks(self, chunks: list[Chunk], *, reset: bool = False) -> None:
        """Index chunks, optionally recreating the target collection first."""


class CVIngestionService:
    """Parse PDFs, chunk them, and index the result into Qdrant."""

    def __init__(
        self,
        *,
        pdf_dir: Path,
        parser: CVDirectoryParser = parse_directory,
        chunker: CVChunker = chunk_cvs,
        indexer: ChunkIndexer | None = None,
    ) -> None:
        """Bind the rendered-PDF source directory and pipeline dependencies."""
        self.pdf_dir = pdf_dir
        self._parser = parser
        self._chunker = chunker
        self._indexer = indexer or QdrantChunkIndexer()

    def ingest(self, *, reset: bool = False) -> IngestionSummary:
        """Run the current ingestion slice from rendered PDFs through Qdrant."""
        parsed_cvs = self._parser(self.pdf_dir)
        if not parsed_cvs:
            return IngestionSummary(pdf_count=0, chunk_count=0, reset=reset)

        chunks = self._chunker(parsed_cvs)
        self._indexer.index_chunks(chunks, reset=reset)
        return IngestionSummary(
            pdf_count=len(parsed_cvs),
            chunk_count=len(chunks),
            reset=reset,
        )
