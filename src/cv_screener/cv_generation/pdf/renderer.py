"""Render YAML CV profiles into PDF documents."""

from pathlib import Path

from jinja2 import Environment, PackageLoader, select_autoescape
from loguru import logger
from weasyprint import HTML

from cv_screener.cv_generation.content.schema import CVProfile
from cv_screener.cv_generation.content.yaml_io import load_cv_profile
from cv_screener.cv_generation.pdf.sections import (
    build_pdf_filename,
    build_section_content,
)
from cv_screener.cv_generation.pdf.templates import select_template


class CVPDFRenderer:
    """Render validated CV profiles into HTML and PDF outputs."""

    def __init__(self) -> None:
        """Initialize the renderer with package-backed templates."""
        self._environment = self._build_environment()

    def render_html(self, profile: CVProfile) -> str:
        """Render a CV profile into HTML using a deterministic template selection."""
        definition, preset = select_template(profile)
        template = self._environment.get_template(f"{definition.template_id}.html.j2")
        sections = build_section_content(profile, definition.section_titles)
        context = {
            "profile": profile,
            "template_name": definition.template_name,
            "template_id": definition.template_id,
            "section_order_preset": preset.preset_id,
            "main_sections": [sections[section_id] for section_id in preset.main_sections],
            "sidebar_sections": [
                sections[section_id] for section_id in preset.sidebar_sections
            ],
            "has_sidebar": bool(preset.sidebar_sections),
            "column_layout": definition.column_layout,
        }
        return template.render(**context)

    def write_pdf(self, profile: CVProfile, output_path: Path) -> None:
        """Render a CV profile and persist the result as a PDF."""
        html = self.render_html(profile)
        self._write_pdf_from_html(html, output_path)

    @staticmethod
    def _build_environment() -> Environment:
        """Create the Jinja environment for package templates."""
        return Environment(
            loader=PackageLoader("cv_screener.cv_generation.pdf"),
            autoescape=select_autoescape(("html", "xml")),
            trim_blocks=True,
            lstrip_blocks=True,
        )

    @staticmethod
    def _write_pdf_from_html(html: str, output_path: Path) -> None:
        """Write rendered HTML to a PDF file with WeasyPrint."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=html).write_pdf(output_path)


class PDFRenderingService:
    """Service for rendering YAML CV files into PDFs."""

    def __init__(
        self,
        *,
        input_dir: Path,
        output_dir: Path,
        renderer: CVPDFRenderer | None = None,
    ) -> None:
        """Initialize the rendering service."""
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.renderer = renderer or CVPDFRenderer()

    def render_directory(self, directory: Path | None = None) -> list[Path]:
        """Render every YAML CV file in a directory into a PDF."""
        source_dir = directory or self.input_dir
        rendered_files: list[Path] = []
        for path in sorted(source_dir.glob("*.yaml")):
            profile = load_cv_profile(path)
            output_path = self.output_dir / build_pdf_filename(profile)
            logger.info(
                "Rendering CV PDF",
                input_path=str(path),
                output_path=str(output_path),
                candidate_id=str(profile.candidate_id),
            )
            self.renderer.write_pdf(profile, output_path)
            rendered_files.append(output_path)
        return rendered_files
