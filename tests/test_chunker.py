"""Tests for the section-aware chunker."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from cv_screener.ingestion import chunker
from cv_screener.ingestion.schema import ChunkConfig, ParsedCV, ParsedPage


def _make_cv(markdown: str, filename: str = "test_cv") -> ParsedCV:
    return ParsedCV(
        source_path=Path(f"/fake/{filename}.pdf"),
        filename=filename,
        pages=[ParsedPage(page_number=1, markdown=markdown)],
    )


def test_chunk_cv_single_section() -> None:
    cv = _make_cv("## **SUMMARY**\n\nA skilled engineer.")
    chunks = chunker.chunk_cv(cv)

    assert len(chunks) == 1
    assert chunks[0].section == "SUMMARY"
    assert "skilled engineer" in chunks[0].text
    assert chunks[0].source_filename == "test_cv"
    assert chunks[0].page_number == 1


def test_chunk_cv_multiple_sections() -> None:
    md = (
        "## **SUMMARY**\n\nA skilled engineer.\n\n"
        "## **EXPERIENCE**\n\nBuilt APIs.\n\n"
        "## **EDUCATION**\n\nBSc Computer Science"
    )
    cv = _make_cv(md)
    chunks = chunker.chunk_cv(cv)

    assert len(chunks) == 3
    sections = [c.section for c in chunks]
    assert sections == ["SUMMARY", "EXPERIENCE", "EDUCATION"]


def test_chunk_cv_name_heading_goes_to_header_section() -> None:
    md = "Marta Alvarez\n\n## MARTA ALVAREZ\n\n## **SUMMARY**\n\nA backend engineer."
    cv = _make_cv(md)
    chunks = chunker.chunk_cv(cv)

    header_chunks = [c for c in chunks if c.section == "header"]
    summary_chunks = [c for c in chunks if c.section == "SUMMARY"]
    assert len(header_chunks) >= 1
    assert len(summary_chunks) == 1


def test_chunk_cv_no_headings() -> None:
    cv = _make_cv("Just some plain text without any headings.")
    chunks = chunker.chunk_cv(cv)

    assert len(chunks) == 1
    assert chunks[0].section == "header"
    assert "plain text" in chunks[0].text


def test_chunk_cv_long_section_stays_as_single_chunk() -> None:
    paragraphs = [f"Paragraph {i} with enough content to fill." for i in range(20)]
    long_text = "\n\n".join(paragraphs)
    md = f"## **EXPERIENCE**\n\n{long_text}"
    cv = _make_cv(md)

    chunks = chunker.chunk_cv(cv)

    experience_chunks = [c for c in chunks if c.section == "EXPERIENCE"]
    assert len(experience_chunks) == 1
    for para in paragraphs:
        assert para in experience_chunks[0].text


def test_chunk_cv_bold_and_plain_headings() -> None:
    md = "## **SUMMARY**\n\nBold heading.\n\n## EXPERIENCE\n\nPlain heading."
    cv = _make_cv(md)
    chunks = chunker.chunk_cv(cv)

    sections = [c.section for c in chunks]
    assert "SUMMARY" in sections
    assert "EXPERIENCE" in sections


def test_chunk_cvs_processes_multiple() -> None:
    cv1 = _make_cv("## **SUMMARY**\n\nEngineer one.", "cv1")
    cv2 = _make_cv("## **SUMMARY**\n\nEngineer two.", "cv2")
    chunks = chunker.chunk_cvs([cv1, cv2])

    assert len(chunks) == 2
    assert chunks[0].source_filename == "cv1"
    assert chunks[1].source_filename == "cv2"


def test_chunk_preserves_multi_page() -> None:
    cv = ParsedCV(
        source_path=Path("/fake/multi.pdf"),
        filename="multi",
        pages=[
            ParsedPage(page_number=1, markdown="## **SUMMARY**\n\nPage one content."),
            ParsedPage(page_number=2, markdown="## **EDUCATION**\n\nPage two content."),
        ],
    )
    chunks = chunker.chunk_cv(cv)

    assert len(chunks) == 2
    assert chunks[0].page_number == 1
    assert chunks[0].section == "SUMMARY"
    assert chunks[1].page_number == 2
    assert chunks[1].section == "EDUCATION"


def test_chunk_config_defaults() -> None:
    config = ChunkConfig()

    assert config.similarity_threshold == 0.6
    assert config.model_name == "BAAI/bge-small-en-v1.5"
    assert "SUMMARY" in config.canonical_sections


def test_chunk_cv_accepts_config() -> None:
    cv = _make_cv("## **SUMMARY**\n\nConfigured chunking.")
    chunks = chunker.chunk_cv(cv, config=ChunkConfig())

    assert len(chunks) == 1
    assert chunks[0].section == "SUMMARY"


def test_chunk_config_validates_similarity_threshold() -> None:
    with pytest.raises(ValidationError, match="similarity_threshold"):
        ChunkConfig(similarity_threshold=1.5)
