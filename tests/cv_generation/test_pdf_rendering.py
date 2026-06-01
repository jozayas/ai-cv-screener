import base64
from pathlib import Path

import pytest

from cv_screener.cv_generation.content.generator import CVGenerationService
from cv_screener.cv_generation.content.yaml_io import load_cv_profile, write_cv_profile
from cv_screener.cv_generation.pdf.renderer import CVPDFRenderer, PDFRenderingService

SAMPLE_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5X"
    "G9QAAAAASUVORK5CYII="
)


def test_render_files_creates_pdf_documents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content_dir = tmp_path / "content"
    pdf_dir = tmp_path / "pdfs"
    content_dir.mkdir()
    pdf_dir.mkdir()
    monkeypatch.chdir(tmp_path)

    generation_service = CVGenerationService(output_dir=content_dir)
    yaml_files = generation_service.generate(count=2)

    renderer = PDFRenderingService(input_dir=content_dir, output_dir=pdf_dir)
    rendered = renderer.render_files(yaml_files)

    assert len(rendered) == 2
    assert all(path.exists() for path in rendered)
    assert all(path.suffix == ".pdf" for path in rendered)


def test_render_html_and_pdf_include_profile_photo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    content_dir = tmp_path / "content"
    pdf_dir = tmp_path / "pdfs"
    photo_dir = tmp_path / "data" / "generated" / "photos"
    content_dir.mkdir()
    pdf_dir.mkdir()
    photo_dir.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    generation_service = CVGenerationService(output_dir=content_dir)
    yaml_path = generation_service.generate(count=1)[0]
    profile = load_cv_profile(yaml_path)
    photo_path = Path("data/generated/photos") / f"{profile.candidate_id}.png"
    (tmp_path / photo_path).write_bytes(SAMPLE_PNG_BYTES)
    profile.photo_path = photo_path.as_posix()
    write_cv_profile(yaml_path, profile)

    renderer = CVPDFRenderer()
    updated_profile = load_cv_profile(yaml_path)
    html = renderer.render_html(updated_profile)
    output_path = pdf_dir / "rendered.pdf"
    renderer.write_pdf(updated_profile, output_path)

    assert photo_path.as_posix() in html
    assert "<img" in html
    assert output_path.exists()
