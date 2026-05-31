"""Lazy CLI dependency builders."""

from __future__ import annotations

from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from cv_screener.cli.serve import ChainlitLauncher
    from cv_screener.cv_generation.pdf.templates import TemplateId
    from cv_screener.ingestion.schema import IngestionSummary
    from cv_screener.rag.service import RAGQueryResult
    from cv_screener.retrieval.schema import RetrievedChunk


class PDFRenderingServiceProtocol(Protocol):
    """Behavior needed from the PDF rendering service in CLI commands."""

    def render_directory(self, directory: Path) -> list[Path]:
        """Render every YAML CV file in a directory into PDFs."""
        ...

    def render_files(self, paths: list[Path]) -> list[Path]:
        """Render the provided YAML CV files into PDFs."""
        ...


class CVIngestionServiceProtocol(Protocol):
    """Behavior needed from the ingestion service in CLI commands."""

    def ingest(
        self,
        *,
        reset: bool = False,
        progress_callback: object | None = None,
        expected_pdf_count: int | None = None,
    ) -> IngestionSummary:
        """Parse, chunk, and index rendered CV PDFs."""
        ...


class HybridRetrieverProtocol(Protocol):
    """Behavior needed from the retriever in CLI commands."""

    def retrieve(self, query_text: str) -> list[RetrievedChunk]:
        """Return ranked chunks for the given query."""
        ...


class RAGQueryServiceProtocol(Protocol):
    """Behavior needed from the RAG runtime in CLI commands."""

    def run(self, query_text: str) -> RAGQueryResult:
        """Return the final RAG response for the given query."""
        ...

    def async_stream(
        self, query_text: str
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield (node_name, state_update) as each graph node completes."""
        ...


def build_pdf_rendering_service(
    *,
    input_dir: Path,
    output_dir: Path,
    template_id: TemplateId | None,
) -> PDFRenderingServiceProtocol:
    """Build the PDF rendering service lazily to keep CLI help fast."""
    module = import_module("cv_screener.cv_generation.pdf.renderer")
    service = module.PDFRenderingService(
        input_dir=input_dir,
        output_dir=output_dir,
        template_id=template_id,
    )
    return cast("PDFRenderingServiceProtocol", service)


def build_cv_ingestion_service(*, pdf_dir: Path) -> CVIngestionServiceProtocol:
    """Build the ingestion service lazily to keep CLI import overhead low."""
    ingestion_module = import_module("cv_screener.ingestion.ingest")
    config_module = import_module("cv_screener.config")
    persistence_module = import_module("cv_screener.persistence")

    sqlite_settings = config_module.SQLiteSettings()
    canonical_store = persistence_module.SQLiteCanonicalRepository(
        sqlite_path=Path(sqlite_settings.sqlite_path),
        content_dir=Path("data/cvs_contents"),
    )
    service = ingestion_module.CVIngestionService(
        pdf_dir=pdf_dir,
        canonical_store=canonical_store,
    )
    return cast("CVIngestionServiceProtocol", service)


def build_hybrid_retriever() -> HybridRetrieverProtocol:
    """Build the hybrid retriever lazily to keep CLI help fast."""
    module = import_module("cv_screener.retrieval.hybrid")
    return cast("HybridRetrieverProtocol", module.HybridRetriever())


def build_rag_query_service() -> RAGQueryServiceProtocol:
    """Build the RAG query service lazily to keep CLI help fast."""
    module = import_module("cv_screener.rag.service")
    return cast("RAGQueryServiceProtocol", module.RAGQueryService())


def build_chainlit_launcher() -> ChainlitLauncher:
    """Build the Chainlit launcher lazily to keep CLI help fast."""
    module = import_module("cv_screener.cli.serve")
    return cast("ChainlitLauncher", module.ChainlitLauncher())
