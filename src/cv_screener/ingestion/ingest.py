"""Orchestration for parsing, chunking, and indexing rendered CV PDFs."""

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from cv_screener.ingestion.chunking import chunk_cvs
from cv_screener.ingestion.chunking.schema import Chunk
from cv_screener.ingestion.indexing import QdrantChunkIndexer
from cv_screener.ingestion.parser import parse_directory
from cv_screener.ingestion.parsing.schema import ParsedCV
from cv_screener.ingestion.schema import IngestionSummary
from cv_screener.persistence import CanonicalStoreProtocol

type ParseProgressCallback = Callable[[int, int], None]
type IngestionProgressCallback = Callable[[int, int, str], None]


class CVDirectoryParser(Protocol):
    """Callable interface for parsing a directory of rendered CV PDFs."""

    def __call__(
        self,
        directory: Path,
        *,
        progress_callback: ParseProgressCallback | None = None,
    ) -> list[ParsedCV]:
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


class CanonicalStore(Protocol):
    """Interface for persisting canonical parsed/extracted CV data."""

    def reset_all(self) -> None:
        """Reset canonical persistence before ingesting a fresh corpus."""

    def persist(
        self, *, parsed_cvs: list[ParsedCV], chunks_to_store: list[Chunk]
    ) -> None:
        """Persist parsed CVs and chunks to the canonical store."""


class CVIngestionService:
    """Parse PDFs, chunk them, and index the result into Qdrant."""

    def __init__(
        self,
        *,
        pdf_dir: Path,
        parser: CVDirectoryParser = parse_directory,
        chunker: CVChunker = chunk_cvs,
        indexer: ChunkIndexer | None = None,
        canonical_store: CanonicalStoreProtocol | None = None,
    ) -> None:
        """Bind the rendered-PDF source directory and pipeline dependencies."""
        self.pdf_dir = pdf_dir
        self._parser = parser
        self._chunker = chunker
        self._indexer = indexer or QdrantChunkIndexer()
        self._canonical_store = canonical_store

    def ingest(
        self,
        *,
        reset: bool = False,
        progress_callback: IngestionProgressCallback | None = None,
        expected_pdf_count: int | None = None,
    ) -> IngestionSummary:
        """Run the current ingestion slice from rendered PDFs through Qdrant."""
        total_steps = self._resolve_total_steps(expected_pdf_count)
        current_step = 0
        self._emit_progress(
            progress_callback, current_step, total_steps, "Parsing PDFs"
        )

        parsed_cvs, current_step = self._parse_cvs(
            progress_callback=progress_callback,
            total_steps=total_steps,
        )
        if not parsed_cvs:
            return IngestionSummary(pdf_count=0, chunk_count=0, reset=reset)

        if total_steps == 0:
            total_steps = len(parsed_cvs) + 3
            current_step = len(parsed_cvs)
        self._emit_progress(
            progress_callback, current_step, total_steps, "Chunking content"
        )
        chunks = self._chunker(parsed_cvs)
        current_step += 1
        self._emit_progress(
            progress_callback, current_step, total_steps, "Persisting SQLite"
        )
        if self._canonical_store is not None:
            if reset:
                self._canonical_store.reset_all()
            self._canonical_store.persist(parsed_cvs=parsed_cvs, chunks_to_store=chunks)
        current_step += 1
        self._emit_progress(
            progress_callback, current_step, total_steps, "Indexing Qdrant"
        )
        self._indexer.index_chunks(chunks, reset=reset)
        current_step += 1
        self._emit_progress(progress_callback, current_step, total_steps, "Finalizing")
        return IngestionSummary(
            pdf_count=len(parsed_cvs),
            chunk_count=len(chunks),
            reset=reset,
        )

    @staticmethod
    def _resolve_total_steps(expected_pdf_count: int | None) -> int:
        if expected_pdf_count is None:
            return 0
        return expected_pdf_count + 3

    @staticmethod
    def _emit_progress(
        progress_callback: IngestionProgressCallback | None,
        current: int,
        total: int,
        stage: str,
    ) -> None:
        if progress_callback is None or total == 0:
            return
        progress_callback(current, total, stage)

    def _parse_cvs(
        self,
        *,
        progress_callback: IngestionProgressCallback | None,
        total_steps: int,
    ) -> tuple[list[ParsedCV], int]:
        if progress_callback is None:
            return self._parser(self.pdf_dir), 0
        try:
            parsed_cvs = self._parser(
                self.pdf_dir,
                progress_callback=lambda current, total: progress_callback(
                    current, total + 3, "Parsing PDFs"
                ),
            )
            return parsed_cvs, len(parsed_cvs)
        except TypeError:
            parsed_cvs = self._parser(self.pdf_dir)
            current = len(parsed_cvs) if total_steps else 0
            return parsed_cvs, current
