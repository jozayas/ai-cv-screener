"""Semantic sub-chunking for section text."""

from dataclasses import dataclass

import numpy as np
from fastembed import TextEmbedding

from cv_screener.ingestion.chunking.common import _SENTENCE_PATTERN
from cv_screener.ingestion.chunking.schema import ChunkConfig
from cv_screener.ingestion.chunking.sectioning import cosine_similarity


def split_semantic_units(text: str) -> list[str]:
    """Split section text into sentence-like semantic units."""
    units = [unit.strip() for unit in _SENTENCE_PATTERN.split(text) if unit.strip()]
    return units or [text.strip()]


@dataclass(frozen=True)
class SectionChunkRequest:
    """Inputs needed to build chunks for a single page section."""

    page_number: int
    starting_chunk_index: int
    section: str
    texts: list[str]


class SemanticChunker:
    """Split text into semantic chunks using adjacent embedding similarity."""

    def __init__(self, config: ChunkConfig, model: TextEmbedding) -> None:
        """Bind semantic chunking thresholds to an embedding model."""
        self._config = config
        self._model = model

    def split(self, text: str) -> list[str]:
        """Chunk section text into semantically coherent windows."""
        units = split_semantic_units(text)
        if len(units) == 1:
            return units

        embeddings = list(self._model.embed(units))
        base_chunks = self._build_base_chunks(units, embeddings)
        return self._apply_overlap(base_chunks)

    def _build_base_chunks(
        self,
        units: list[str],
        embeddings: list[np.ndarray],
    ) -> list[list[str]]:
        chunks: list[list[str]] = []
        current_units = [units[0]]
        current_embeddings = [embeddings[0]]

        for index in range(1, len(units)):
            unit = units[index]
            unit_embedding = embeddings[index]
            candidate_text = " ".join([*current_units, unit])
            average_embedding = np.mean(np.vstack(current_embeddings), axis=0)
            similarity = cosine_similarity(average_embedding, unit_embedding)
            should_split = len(candidate_text) > self._config.chunk_size or (
                similarity < self._config.semantic_similarity_threshold
                and len(" ".join(current_units)) >= (self._config.chunk_size // 3)
            )

            if should_split:
                chunks.append(current_units)
                current_units = [unit]
                current_embeddings = [unit_embedding]
                continue

            current_units.append(unit)
            current_embeddings.append(unit_embedding)

        chunks.append(current_units)
        return chunks

    def _apply_overlap(self, chunks: list[list[str]]) -> list[str]:
        overlapped_chunks: list[str] = []
        previous_units: list[str] = []

        for chunk_units in chunks:
            prefix_units: list[str] = []
            if previous_units and self._config.chunk_overlap > 0:
                overlap_length = 0
                for unit in reversed(previous_units):
                    prefix_units.insert(0, unit)
                    overlap_length = len(" ".join(prefix_units))
                    if overlap_length >= self._config.chunk_overlap:
                        break

            overlapped_chunks.append(" ".join([*prefix_units, *chunk_units]).strip())
            previous_units = chunk_units

        return overlapped_chunks
