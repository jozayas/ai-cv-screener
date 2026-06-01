"""Public chunking API for parsed CV documents."""

from dataclasses import dataclass

from loguru import logger

from cv_screener.ingestion.chunking.metadata import MetadataExtractor
from cv_screener.ingestion.chunking.schema import Chunk, ChunkConfig
from cv_screener.ingestion.chunking.sectioning import (
    MARKDOWN_SPLITTER,
    SectionClassifier,
    resolve_section,
)
from cv_screener.ingestion.chunking.semantic import SectionChunkRequest, SemanticChunker
from cv_screener.ingestion.parsing.schema import ParsedCV


@dataclass(frozen=True)
class _ChunkBuildContext:
    cv: ParsedCV
    provenance: "_ChunkProvenance"
    candidate_name: str | None
    metadata_extractor: MetadataExtractor


@dataclass(frozen=True)
class _ChunkProvenance:
    source_file: str
    document_title: str


@dataclass(frozen=True)
class _ChunkingRuntime:
    classifier: SectionClassifier
    metadata_extractor: MetadataExtractor
    semantic_chunker: SemanticChunker


def _build_runtime(config: ChunkConfig) -> _ChunkingRuntime:
    classifier = SectionClassifier(config)
    metadata_extractor = MetadataExtractor(config)
    semantic_chunker = SemanticChunker(config, classifier.model)
    return _ChunkingRuntime(
        classifier=classifier,
        metadata_extractor=metadata_extractor,
        semantic_chunker=semantic_chunker,
    )


def _build_chunk(
    *,
    context: _ChunkBuildContext,
    page: int,
    chunk_index: int,
    section: str,
    text: str,
) -> Chunk | None:
    cleaned_text = text.strip()
    if not cleaned_text:
        return None

    extraction = context.metadata_extractor.extract_chunk_metadata(cleaned_text)

    return Chunk(
        candidate_name=context.candidate_name,
        source_file=context.provenance.source_file,
        document_title=context.provenance.document_title,
        page=page,
        chunk_index=chunk_index,
        section=section,
        detected_skills=extraction.detected_skills,
        detected_companies=extraction.detected_companies,
        detected_universities=extraction.detected_universities,
        email_addresses=extraction.email_addresses,
        phone_numbers=extraction.phone_numbers,
        linkedin_urls=extraction.linkedin_urls,
        github_urls=extraction.github_urls,
        text=cleaned_text,
    )


def _build_provenance(cv: ParsedCV) -> _ChunkProvenance:
    return _ChunkProvenance(
        source_file=cv.source_path.name,
        document_title=cv.title.strip(),
    )


def _build_section_chunks(
    *,
    context: _ChunkBuildContext,
    request: SectionChunkRequest,
    semantic_chunker: SemanticChunker,
) -> list[Chunk]:
    combined = "\n\n".join(text for text in request.texts if text).strip()
    if not combined:
        return []

    chunks: list[Chunk] = []
    next_chunk_index = request.starting_chunk_index
    for text in semantic_chunker.split(combined):
        chunk = _build_chunk(
            context=context,
            page=request.page_number,
            chunk_index=next_chunk_index,
            section=request.section,
            text=text,
        )
        if chunk is not None:
            chunks.append(chunk)
            next_chunk_index += 1
    return chunks


def chunk_cv(
    cv: ParsedCV,
    config: ChunkConfig | None = None,
    *,
    runtime: _ChunkingRuntime | None = None,
) -> list[Chunk]:
    """Split a parsed CV into retrieval-ready semantic chunks."""
    resolved_config = config or ChunkConfig()
    resolved_runtime = runtime or _build_runtime(resolved_config)
    context = _ChunkBuildContext(
        cv=cv,
        provenance=_build_provenance(cv),
        candidate_name=resolved_runtime.metadata_extractor.candidate_name(cv),
        metadata_extractor=resolved_runtime.metadata_extractor,
    )
    chunks: list[Chunk] = []
    next_chunk_index = 0

    for page in cv.pages:
        page_by_section: dict[str, list[str]] = {}
        for doc in MARKDOWN_SPLITTER.split_text(page.markdown):
            section = resolve_section(
                doc.metadata.get("section", ""),
                resolved_runtime.classifier,
            )
            page_by_section.setdefault(section, []).append(doc.page_content.strip())

        for section, texts in page_by_section.items():
            section_chunks = _build_section_chunks(
                context=context,
                request=SectionChunkRequest(
                    page_number=page.page_number,
                    starting_chunk_index=next_chunk_index,
                    section=section,
                    texts=texts,
                ),
                semantic_chunker=resolved_runtime.semantic_chunker,
            )
            chunks.extend(section_chunks)
            next_chunk_index += len(section_chunks)

    logger.debug(
        "Chunked CV",
        filename=cv.filename,
        pages=len(cv.pages),
        chunks=len(chunks),
    )
    return chunks


def chunk_cvs(cvs: list[ParsedCV], config: ChunkConfig | None = None) -> list[Chunk]:
    """Split multiple parsed CVs into section-aware chunks."""
    resolved_config = config or ChunkConfig()
    runtime = _build_runtime(resolved_config)
    return [
        chunk
        for cv in cvs
        for chunk in chunk_cv(cv, config=resolved_config, runtime=runtime)
    ]
