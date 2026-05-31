from pathlib import Path

from cv_screener.cv_generation.content.generator import CVGenerationService
from cv_screener.cv_generation.pdf.renderer import PDFRenderingService


def test_render_files_creates_pdf_documents(tmp_path: Path) -> None:
    content_dir = tmp_path / "content"
    pdf_dir = tmp_path / "pdfs"
    content_dir.mkdir()
    pdf_dir.mkdir()

    generation_service = CVGenerationService(output_dir=content_dir)
    yaml_files = generation_service.generate(count=2)

    renderer = PDFRenderingService(input_dir=content_dir, output_dir=pdf_dir)
    rendered = renderer.render_files(yaml_files)

    assert len(rendered) == 2
    assert all(path.exists() for path in rendered)
    assert all(path.suffix == ".pdf" for path in rendered)
