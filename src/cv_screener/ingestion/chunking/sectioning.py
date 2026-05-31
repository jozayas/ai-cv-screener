"""Heading-based section classification for parsed CV content."""

import contextlib
import functools
import io
import logging
import os
import warnings

import numpy as np
from fastembed import TextEmbedding
from langchain_text_splitters import MarkdownHeaderTextSplitter

from cv_screener.ingestion.chunking.common import _HEADER_SECTION
from cv_screener.ingestion.chunking.schema import ChunkConfig

MARKDOWN_SPLITTER = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("##", "section")],
)


@functools.lru_cache(maxsize=1)
def _configure_hf_noise() -> None:
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
    warnings.filterwarnings(
        "ignore",
        message="The `resume_download` argument is deprecated and ignored.*",
        category=UserWarning,
    )


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    denominator = np.linalg.norm(left) * np.linalg.norm(right)
    if denominator == 0:
        return 0.0
    return float(np.dot(left, right) / denominator)


class SectionClassifier:
    """Lazy-loaded classifier that maps headings to canonical CV sections."""

    def __init__(self, config: ChunkConfig) -> None:
        """Create a classifier backed by the configured embedding model."""
        self._config = config
        self._model: TextEmbedding | None = None
        self._section_vectors: np.ndarray | None = None

    @property
    def model(self) -> TextEmbedding:
        """Return the lazily initialized embedding model."""
        if self._model is None:
            _configure_hf_noise()
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                self._model = TextEmbedding(model_name=self._config.model_name)
        return self._model

    def section_vectors(self) -> np.ndarray:
        """Embed configured canonical sections once and reuse them."""
        if self._section_vectors is None:
            _configure_hf_noise()
            with (
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(io.StringIO()),
            ):
                embeddings = np.vstack(
                    list(self.model.embed(self._config.canonical_sections))
                )
            self._section_vectors = embeddings
        return self._section_vectors

    def classify(self, raw_section: str) -> str:
        """Map a raw heading to the closest canonical section."""
        cleaned = raw_section.strip().strip("*").strip().upper()
        if not cleaned:
            return _HEADER_SECTION

        if cleaned in self._config.canonical_sections:
            return cleaned

        _configure_hf_noise()
        with (
            contextlib.redirect_stdout(io.StringIO()),
            contextlib.redirect_stderr(io.StringIO()),
        ):
            heading_vector = next(iter(self.model.embed([cleaned])))
        similarities = [
            cosine_similarity(section_vector, heading_vector)
            for section_vector in self.section_vectors()
        ]
        best_match = int(np.argmax(similarities).item())

        if similarities[best_match] >= self._config.similarity_threshold:
            return self._config.canonical_sections[best_match]
        return _HEADER_SECTION


def resolve_section(doc_section: str, classifier: SectionClassifier) -> str:
    """Resolve a markdown heading into a canonical section label."""
    if not doc_section:
        return _HEADER_SECTION
    return classifier.classify(doc_section)
