"""Tests for the section-aware semantic chunker."""

from pathlib import Path

import pytest
from fastembed import TextEmbedding
from pydantic import ValidationError

from cv_screener.ingestion.chunking import chunker
from cv_screener.ingestion.chunking.schema import ChunkConfig
from cv_screener.ingestion.parsing.schema import ParsedCV, ParsedPage


def _make_cv(
    markdown: str,
    filename: str = "external-cv",
    title: str = "",
) -> ParsedCV:
    return ParsedCV(
        source_path=Path(f"/fake/{filename}.pdf"),
        filename=filename,
        title=title,
        pages=[ParsedPage(page_number=1, markdown=markdown)],
    )


@pytest.fixture(autouse=True)
def fake_metadata_extraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_predict_entities(
        _: object,
        text: str,
        labels: list[str],
    ) -> list[dict[str, str | float]]:
        lowered = text.lower()
        entities: list[dict[str, str | float]] = []
        if "person" in labels and "marta alvarez" in lowered:
            entities.append({"text": "Marta Alvarez", "label": "person", "score": 0.99})
        if "skill" in labels:
            if "python" in lowered:
                entities.append({"text": "Python", "label": "skill", "score": 0.97})
            if "fastapi" in lowered:
                entities.append({"text": "FastAPI", "label": "skill", "score": 0.95})
        if "company" in labels and "novastack" in lowered:
            entities.append({"text": "NovaStack", "label": "company", "score": 0.96})
        if "university" in labels and "universidad complutense" in lowered:
            entities.append(
                {
                    "text": "Universidad Complutense",
                    "label": "university",
                    "score": 0.94,
                }
            )
        return entities

    monkeypatch.setattr(
        chunker.MetadataExtractor,
        "_predict_entities",
        fake_predict_entities,
    )


def test_chunk_cv_single_section() -> None:
    cv = _make_cv(
        "Marta Alvarez\nmarta@example.com\nhttps://linkedin.com/in/marta\n"
        "## **SUMMARY**\n\nBuilds resilient Python and FastAPI APIs.",
        title="Marta Alvarez CV",
    )
    chunks = chunker.chunk_cv(cv)
    summary_chunk = next(chunk for chunk in chunks if chunk.section == "SUMMARY")
    header_chunk = next(chunk for chunk in chunks if chunk.section == "header")

    assert len(chunks) == 2
    assert summary_chunk.section == "SUMMARY"
    assert summary_chunk.text == "Builds resilient Python and FastAPI APIs."
    assert summary_chunk.source_file == "external-cv.pdf"
    assert summary_chunk.document_title == "Marta Alvarez CV"
    assert summary_chunk.page == 1
    assert summary_chunk.chunk_index == 1
    assert summary_chunk.candidate_name == "Marta Alvarez"
    assert summary_chunk.detected_skills == ["Python", "FastAPI"]
    assert header_chunk.email_addresses == ["marta@example.com"]
    assert header_chunk.linkedin_urls == ["https://linkedin.com/in/marta"]


def test_chunk_cv_multiple_sections() -> None:
    md = (
        "## **SUMMARY**\n\nA skilled engineer.\n\n"
        "## **EXPERIENCE**\n\nBuilt APIs at NovaStack.\n\n"
        "## **EDUCATION**\n\nBSc Computer Science at Universidad Complutense"
    )
    cv = _make_cv(md)
    chunks = chunker.chunk_cv(cv)

    assert len(chunks) == 3
    sections = [c.section for c in chunks]
    assert sections == ["SUMMARY", "EXPERIENCE", "EDUCATION"]
    assert chunks[1].detected_companies == ["NovaStack"]
    assert chunks[2].detected_universities == ["Universidad Complutense"]


def test_chunk_cv_preserves_header_chunks() -> None:
    md = "Marta Alvarez\n\n## MARTA ALVAREZ\n\n## **SUMMARY**\n\nA backend engineer."
    cv = _make_cv(md)
    chunks = chunker.chunk_cv(cv)

    header_chunks = [c for c in chunks if c.section == "header"]
    summary_chunks = [c for c in chunks if c.section == "SUMMARY"]
    assert len(header_chunks) >= 1
    assert len(summary_chunks) == 1


def test_chunk_cv_no_headings_returns_header_chunk() -> None:
    cv = _make_cv("Just some plain text without any headings.")
    chunks = chunker.chunk_cv(cv)

    assert len(chunks) == 1
    assert chunks[0].section == "header"
    assert "plain text" in chunks[0].text
    assert chunks[0].candidate_name is None


def test_chunk_cv_long_section_is_semantically_subchunked() -> None:
    paragraphs = [f"Python platform delivery sentence {i}." for i in range(10)]
    long_text = "\n\n".join(paragraphs)
    md = f"## **EXPERIENCE**\n\n{long_text}"
    cv = _make_cv(md)

    chunks = chunker.chunk_cv(
        cv,
        config=ChunkConfig(
            chunk_size=120,
            chunk_overlap=25,
            semantic_similarity_threshold=0.0,
        ),
    )

    experience_chunks = [c for c in chunks if c.section == "EXPERIENCE"]
    assert len(experience_chunks) > 1
    combined_text = "\n".join(chunk.text for chunk in experience_chunks)
    for para in paragraphs:
        assert para in combined_text
    assert "sentence 2." in experience_chunks[1].text


def test_chunk_cv_bold_and_plain_headings() -> None:
    md = "## **SUMMARY**\n\nBold heading.\n\n## EXPERIENCE\n\nPlain heading."
    cv = _make_cv(md)
    chunks = chunker.chunk_cv(cv)

    sections = [c.section for c in chunks]
    assert "SUMMARY" in sections
    assert "EXPERIENCE" in sections


def test_chunk_cv_uses_classifier_output_as_canonical_section(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_classifier = chunker.SectionClassifier

    class CanonicalizingSectionClassifier(original_classifier):
        def classify(self, raw_section: str) -> str:
            if "professional experience" in raw_section.casefold():
                return "EXPERIENCE"
            return super().classify(raw_section)

    monkeypatch.setattr(
        chunker,
        "SectionClassifier",
        CanonicalizingSectionClassifier,
    )

    cv = _make_cv("## Professional Experience\n\nBuilt APIs.")

    chunks = chunker.chunk_cv(cv)

    assert len(chunks) == 1
    assert chunks[0].section == "EXPERIENCE"


def test_chunk_cvs_processes_multiple() -> None:
    cv1 = _make_cv("## **SUMMARY**\n\nEngineer one.", "engineer-one")
    cv2 = _make_cv("## **SUMMARY**\n\nEngineer two.", "engineer-two")
    chunks = chunker.chunk_cvs([cv1, cv2])

    assert len(chunks) == 2
    assert chunks[0].source_file == "engineer-one.pdf"
    assert chunks[1].source_file == "engineer-two.pdf"


def test_chunk_cvs_reuses_runtime_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cv1 = _make_cv("## **SUMMARY**\n\nEngineer one.", "engineer-one")
    cv2 = _make_cv("## **SUMMARY**\n\nEngineer two.", "engineer-two")
    calls = {"classifier": 0, "metadata": 0, "semantic": 0}

    original_classifier = chunker.SectionClassifier
    original_metadata = chunker.MetadataExtractor
    original_semantic = chunker.SemanticChunker

    class CountingSectionClassifier(original_classifier):
        def __init__(self, config: ChunkConfig) -> None:
            calls["classifier"] += 1
            super().__init__(config)

    class CountingMetadataExtractor(original_metadata):
        def __init__(self, config: ChunkConfig) -> None:
            calls["metadata"] += 1
            super().__init__(config)

    class CountingSemanticChunker(original_semantic):
        def __init__(self, config: ChunkConfig, model: TextEmbedding) -> None:
            calls["semantic"] += 1
            super().__init__(config, model)

    monkeypatch.setattr(chunker, "SectionClassifier", CountingSectionClassifier)
    monkeypatch.setattr(chunker, "MetadataExtractor", CountingMetadataExtractor)
    monkeypatch.setattr(chunker, "SemanticChunker", CountingSemanticChunker)

    _ = chunker.chunk_cvs([cv1, cv2])

    assert calls == {"classifier": 1, "metadata": 1, "semantic": 1}


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
    assert chunks[0].page == 1
    assert chunks[0].section == "SUMMARY"
    assert chunks[1].page == 2
    assert chunks[1].section == "EDUCATION"
    assert [chunk.chunk_index for chunk in chunks] == [0, 1]


def test_chunk_config_defaults() -> None:
    config = ChunkConfig()

    assert config.similarity_threshold == 0.6
    assert config.model_name == "BAAI/bge-small-en-v1.5"
    assert "SUMMARY" in config.canonical_sections
    assert config.chunk_size == 600
    assert config.chunk_overlap == 80
    assert config.gliner_model_name == "urchade/gliner_small-v2.1"
    assert config.gliner_threshold == 0.5


def test_chunk_cv_accepts_config() -> None:
    cv = _make_cv("## **SUMMARY**\n\nConfigured chunking.")
    chunks = chunker.chunk_cv(cv, config=ChunkConfig())

    assert len(chunks) == 1
    assert chunks[0].section == "SUMMARY"


def test_chunk_config_validates_similarity_threshold() -> None:
    with pytest.raises(ValidationError, match="similarity_threshold"):
        ChunkConfig(similarity_threshold=1.5)


def test_chunk_config_validates_chunk_overlap() -> None:
    with pytest.raises(ValidationError, match="chunk_overlap"):
        ChunkConfig(chunk_size=100, chunk_overlap=100)


def test_chunk_candidate_name_falls_back_to_pdf_title() -> None:
    cv = _make_cv(
        "## **SUMMARY**\n\nExperienced engineer.",
        title="Fatima Noor",
    )

    chunks = chunker.chunk_cv(cv)

    assert chunks[0].candidate_name is None


def test_chunk_candidate_name_is_optional_for_external_cvs() -> None:
    cv = _make_cv("## **SUMMARY**\n\nExperienced engineer.")

    chunks = chunker.chunk_cv(cv)

    assert chunks[0].candidate_name is None


def test_chunk_extracts_contact_links() -> None:
    cv = _make_cv(
        "## **SUMMARY**\n\nReach me at jane@example.com or +34 600 123 123.\n"
        "GitHub: https://github.com/janedoe",
    )

    chunks = chunker.chunk_cv(cv)

    assert chunks[0].email_addresses == ["jane@example.com"]
    assert chunks[0].phone_numbers == ["+34600123123"]
    assert chunks[0].github_urls == ["https://github.com/janedoe"]
