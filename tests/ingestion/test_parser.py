"""Tests for the PDF parser."""

from pathlib import Path

import pytest

from cv_screener.cv_generation.content.generator import CVGenerationService
from cv_screener.cv_generation.pdf.renderer import PDFRenderingService
from cv_screener.ingestion.parser import parse_directory, parse_pdf
from cv_screener.ingestion.parsing.schema import ParsedCV


@pytest.fixture
def pdf_directory(tmp_path: Path) -> Path:
    """Generate and render 3 CV PDFs into a temporary directory."""
    content_dir = tmp_path / "yaml"
    pdf_dir = tmp_path / "pdfs"
    content_dir.mkdir()
    pdf_dir.mkdir()
    service = CVGenerationService(output_dir=content_dir)
    yaml_files = service.generate(count=3)
    renderer = PDFRenderingService(input_dir=content_dir, output_dir=pdf_dir)
    renderer.render_files(yaml_files)
    return pdf_dir


def test_parse_pdf_extracts_text_from_real_pdf(pdf_directory: Path) -> None:
    pdfs = sorted(pdf_directory.glob("*.pdf"))
    assert len(pdfs) >= 1

    parsed = parse_pdf(pdfs[0])

    assert isinstance(parsed, ParsedCV)
    assert parsed.filename == pdfs[0].stem
    assert parsed.source_path == pdfs[0]
    assert len(parsed.pages) >= 1
    assert len(parsed.full_text) > 0


def test_parse_pdf_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        parse_pdf(tmp_path / "nonexistent.pdf")


def test_parse_pdf_raises_for_non_pdf(tmp_path: Path) -> None:
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("hello")

    with pytest.raises(ValueError, match=r"Expected a \.pdf file"):
        parse_pdf(txt_file)


def test_parse_directory_processes_all_pdfs(pdf_directory: Path) -> None:
    results = parse_directory(pdf_directory)

    assert len(results) == 3
    assert all(isinstance(r, ParsedCV) for r in results)
    assert all(len(r.pages) >= 1 for r in results)
    assert all(len(r.full_text) > 0 for r in results)


def test_parse_directory_raises_for_non_directory(tmp_path: Path) -> None:
    fake_dir = tmp_path / "not_a_dir"
    with pytest.raises(NotADirectoryError):
        parse_directory(fake_dir)


def test_parse_directory_returns_empty_for_empty_directory(tmp_path: Path) -> None:
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    results = parse_directory(empty_dir)

    assert results == []


def test_parse_directory_reports_progress(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    pdf_paths = [pdf_dir / f"cv-{idx}.pdf" for idx in range(3)]
    for path in pdf_paths:
        path.write_bytes(b"%PDF-1.4")

    def fake_parse_pdf(pdf_path: Path) -> ParsedCV:
        return ParsedCV.model_validate(
            {
                "source_path": pdf_path,
                "filename": pdf_path.stem,
                "title": "",
                "pages": [{"page_number": 1, "markdown": "Sample"}],
            }
        )

    monkeypatch.setattr("cv_screener.ingestion.parser.parse_pdf", fake_parse_pdf)

    updates: list[tuple[int, int]] = []
    results = parse_directory(
        pdf_dir,
        progress_callback=lambda current, total: updates.append((current, total)),
    )

    assert len(results) == 3
    assert updates[-1] == (3, 3)
    assert len(updates) == 3


def test_parsed_cv_full_text_concatenates_pages(pdf_directory: Path) -> None:
    pdfs = sorted(pdf_directory.glob("*.pdf"))
    parsed = parse_pdf(pdfs[0])

    combined = parsed.full_text
    assert len(combined) > 0
    for page in parsed.pages:
        assert page.markdown in combined
