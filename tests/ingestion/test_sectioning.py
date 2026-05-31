"""Tests for section classification in ingestion."""

import pytest

from cv_screener.ingestion.chunking import sectioning as sectioning_module
from cv_screener.ingestion.chunking.schema import ChunkConfig
from cv_screener.ingestion.chunking.sectioning import SectionClassifier


def test_section_classifier_returns_canonical_matches_without_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ExplodingEmbedding:
        def __init__(self, **_: object) -> None:
            msg = "embedding model should not be loaded"
            raise AssertionError(msg)

    monkeypatch.setattr(sectioning_module, "TextEmbedding", ExplodingEmbedding)

    classifier = SectionClassifier(ChunkConfig())

    assert classifier.classify("SUMMARY") == "SUMMARY"
    assert classifier.classify("**EXPERIENCE**") == "EXPERIENCE"
