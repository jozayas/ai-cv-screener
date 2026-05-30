"""Section-aware chunker for parsed CV documents.

Uses LangChain's MarkdownHeaderTextSplitter to split by markdown headings
and FastEmbed embedding similarity to classify sections. Semantic sub-chunking
will be added in a later slice.
"""

import numpy as np
from fastembed import TextEmbedding
from langchain_text_splitters import MarkdownHeaderTextSplitter
from loguru import logger

from cv_screener.ingestion.schema import Chunk, ChunkConfig, ParsedCV

_HEADER_SECTION = "header"

_MARKDOWN_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("##", "section")],
)


class _SectionClassifier:
    """Lazy-loaded classifier that maps headings to canonical CV sections."""

    def __init__(self, config: ChunkConfig) -> None:
        self._config = config
        self._model: TextEmbedding | None = None
        self._section_vectors: np.ndarray | None = None

    @property
    def model(self) -> TextEmbedding:
        if self._model is None:
            self._model = TextEmbedding(model_name=self._config.model_name)
        return self._model

    def section_vectors(self) -> np.ndarray:
        if self._section_vectors is None:
            embeddings = np.vstack(
                list(self.model.embed(self._config.canonical_sections))
            )
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            vectors = embeddings / norms
            self._section_vectors = vectors
            return vectors
        return self._section_vectors

    @property
    def canonical_sections(self) -> list[str]:
        return self._config.canonical_sections

    @property
    def similarity_threshold(self) -> float:
        return self._config.similarity_threshold

    def classify(self, raw_section: str) -> str:
        """Map a raw heading to the closest canonical section."""
        cleaned = raw_section.strip().strip("*").strip().upper()
        heading_vector = next(iter(self.model.embed([cleaned])))
        heading_vector = heading_vector / np.linalg.norm(heading_vector)

        similarities = self.section_vectors() @ heading_vector
        best_match = np.argmax(similarities).item()

        if similarities[best_match].item() >= self.similarity_threshold:
            return self.canonical_sections[best_match]
        return _HEADER_SECTION


def _resolve_section(doc_section: str, classifier: _SectionClassifier) -> str:
    if not doc_section:
        return _HEADER_SECTION
    return classifier.classify(doc_section)


def _build_chunk(
    *, cv: ParsedCV, page_number: int, section: str, texts: list[str]
) -> Chunk | None:
    combined = "\n\n".join(text for text in texts if text).strip()
    if not combined:
        return None
    return Chunk(
        source_filename=cv.filename,
        page_number=page_number,
        section=section,
        text=combined,
    )


def chunk_cv(cv: ParsedCV, config: ChunkConfig | None = None) -> list[Chunk]:
    """Split a parsed CV into section-aware chunks based on markdown headings."""
    classifier = _SectionClassifier(config or ChunkConfig())
    chunks: list[Chunk] = []

    for page in cv.pages:
        page_by_section: dict[str, list[str]] = {}
        for doc in _MARKDOWN_SPLITTER.split_text(page.markdown):
            section = _resolve_section(doc.metadata.get("section", ""), classifier)
            page_by_section.setdefault(section, []).append(doc.page_content.strip())

        for section, texts in page_by_section.items():
            chunk = _build_chunk(
                cv=cv,
                page_number=page.page_number,
                section=section,
                texts=texts,
            )
            if chunk is not None:
                chunks.append(chunk)

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
    all_chunks = [chunk for cv in cvs for chunk in chunk_cv(cv, config=resolved_config)]
    logger.info("Chunked all CVs", total_cvs=len(cvs), total_chunks=len(all_chunks))
    return all_chunks
