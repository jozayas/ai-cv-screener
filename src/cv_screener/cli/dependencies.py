"""Lazy CLI dependency builders."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any, Protocol, cast

if TYPE_CHECKING:
    from collections.abc import AsyncIterator
    from pathlib import Path

    from cv_screener.cli.serve import ChainlitLauncher
    from cv_screener.config import GenerationConfig, QdrantConfig, RAGConfig
    from cv_screener.cv_generation.pdf.templates import TemplateId
    from cv_screener.cv_generation.photos.service import PhotoGenerationSummary
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


class PhotoGenerationServiceProtocol(Protocol):
    """Behavior needed from the CV photo generation service in CLI commands."""

    def generate_directory(self, directory: Path) -> PhotoGenerationSummary:
        """Generate photos for every YAML profile in a directory."""
        ...

    def generate_files(self, paths: list[Path]) -> PhotoGenerationSummary:
        """Generate photos for the provided YAML CV files."""
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
        self, query_text: str, *, conversation_context: str | None = None
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


def build_photo_generation_service(
    *, photo_dir: Path, generation_settings: GenerationConfig
) -> PhotoGenerationServiceProtocol:
    """Build the photo generation service lazily to keep CLI help fast."""
    module = import_module("cv_screener.cv_generation.photos.service")
    service = module.CVPhotoGenerationService(
        photo_dir=photo_dir,
        settings=generation_settings,
    )
    return cast("PhotoGenerationServiceProtocol", service)


def build_cv_ingestion_service(
    *,
    pdf_dir: Path,
    sqlite_path: Path,
    content_dir: Path,
    qdrant_settings: QdrantConfig,
) -> CVIngestionServiceProtocol:
    """Build the ingestion service lazily to keep CLI import overhead low."""
    ingestion_module = import_module("cv_screener.ingestion.ingest")
    qdrant_module = import_module("cv_screener.ingestion.indexing.qdrant")
    schema_module = import_module("cv_screener.ingestion.indexing.schema")
    repository_module = import_module("cv_screener.persistence.repository")
    canonical_store = repository_module.SQLiteCanonicalRepository(
        sqlite_path=sqlite_path,
        content_dir=content_dir,
    )
    service = ingestion_module.CVIngestionService(
        pdf_dir=pdf_dir,
        indexer=qdrant_module.QdrantChunkIndexer(
            config=schema_module.QdrantIndexConfig(
                url=qdrant_settings.qdrant_url,
                check_compatibility=qdrant_settings.qdrant_check_compatibility,
            )
        ),
        canonical_store=canonical_store,
    )
    return cast("CVIngestionServiceProtocol", service)


def build_hybrid_retriever(*, qdrant_settings: QdrantConfig) -> HybridRetrieverProtocol:
    """Build the hybrid retriever lazily to keep CLI help fast."""
    module = import_module("cv_screener.retrieval.hybrid")
    schema_module = import_module("cv_screener.retrieval.schema")
    return cast(
        "HybridRetrieverProtocol",
        module.HybridRetriever(
            config=schema_module.HybridRetrievalConfig(url=qdrant_settings.qdrant_url)
        ),
    )


def build_rag_query_service(
    *,
    rag_settings: RAGConfig,
    sqlite_path: Path,
    candidate_name_min_score: float,
    qdrant_settings: QdrantConfig,
) -> RAGQueryServiceProtocol:
    """Build the RAG query service lazily to keep CLI help fast."""
    module = import_module("cv_screener.rag.service")
    return cast(
        "RAGQueryServiceProtocol",
        module.RAGQueryService(
            dependencies=module.build_graph_dependencies(
                rag_settings=rag_settings,
                sqlite_path=sqlite_path,
                candidate_name_min_score=candidate_name_min_score,
                qdrant_url=qdrant_settings.qdrant_url,
            )
        ),
    )


def build_chainlit_launcher() -> ChainlitLauncher:
    """Build the Chainlit launcher lazily to keep CLI help fast."""
    module = import_module("cv_screener.cli.serve")
    return cast("ChainlitLauncher", module.ChainlitLauncher())
