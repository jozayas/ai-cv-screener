"""Typer application and command definitions."""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer

from cv_screener.cli.generation import generate_cv_content_files
from cv_screener.cli.runtime import (
    ColorMode,
    LogLevel,
    configure_callback,
    init_command,
)
from cv_screener.cv_generation.content.generator import (
    CVGenerationService,
    GenerationMode,
)
from cv_screener.cv_generation.pdf.templates import TemplateId
from cv_screener.ingestion import IngestionSummary

if TYPE_CHECKING:
    from cv_screener.cv_generation.pdf.renderer import PDFRenderingService
    from cv_screener.ingestion import CVIngestionService

app = typer.Typer(
    help="CV screener CLI.",
    no_args_is_help=True,
)


@app.callback()
def main_callback(
    ctx: typer.Context,
    log_level: Annotated[
        LogLevel,
        typer.Option(help="Global log level for CLI commands."),
    ] = LogLevel.INFO,
    no_progress: Annotated[  # noqa: FBT002
        bool,
        typer.Option(
            "--no-progress",
            help="Disable progress bars and similar interactive status output.",
        ),
    ] = False,
    color: Annotated[
        ColorMode,
        typer.Option(help="Color mode for CLI output."),
    ] = ColorMode.AUTO,
) -> None:
    """CV screener CLI."""
    configure_callback(
        ctx=ctx,
        log_level=log_level,
        no_progress=no_progress,
        color=color,
    )


@app.command("generate-content")
def generate_content(
    ctx: typer.Context,
    count: int = typer.Option(3, min=1, max=50, help="Number of YAML CVs to generate."),
    mode: Annotated[
        GenerationMode,
        typer.Option(
            help="Generation mode: seeded local samples or LLM-backed generation."
        ),
    ] = GenerationMode.SEEDED,
    output_dir: Path = typer.Option(
        Path("data/cvs_contents"),
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Directory where YAML CVs will be written.",
    ),
) -> list[Path]:
    """Generate validated CV content YAML files."""
    written_files = generate_cv_content_files(
        ctx=ctx,
        count=count,
        mode=mode,
        output_dir=output_dir,
    )
    typer.echo(f"Generated {len(written_files)} CV YAML files in {output_dir}.")
    return written_files


@app.command("generate-cvs")
def generate_cvs(  # noqa: PLR0913
    ctx: typer.Context,
    count: int = typer.Option(
        3, min=1, max=50, help="Number of CVs to generate and render."
    ),
    mode: Annotated[
        GenerationMode,
        typer.Option(
            help="Generation mode: seeded local samples or LLM-backed generation."
        ),
    ] = GenerationMode.SEEDED,
    content_dir: Path = typer.Option(
        Path("data/cvs_contents"),
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Directory where YAML CVs will be written.",
    ),
    pdf_dir: Path = typer.Option(
        Path("data/cv_pdfs"),
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Directory where PDF CVs will be written.",
    ),
    template_id: Annotated[
        TemplateId | None,
        typer.Option(
            help="Optional PDF template to use instead of deterministic selection."
        ),
    ] = None,
) -> None:
    """Generate validated CV content YAML files and render them into PDFs."""
    written_files = generate_cv_content_files(
        ctx=ctx,
        count=count,
        mode=mode,
        output_dir=content_dir,
    )
    typer.echo(f"Generated {len(written_files)} CV YAML files in {content_dir}.")
    rendered_files = build_pdf_rendering_service(
        input_dir=content_dir,
        output_dir=pdf_dir,
        template_id=template_id,
    ).render_files(written_files)
    typer.echo(f"Rendered {len(rendered_files)} CV PDFs in {pdf_dir}.")


@app.command("validate")
def validate(
    ctx: typer.Context,
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            help="YAML CV file or directory of YAML CV files to validate.",
        ),
    ],
) -> None:
    """Validate one YAML CV file or a directory of YAML CV files."""
    init_command(ctx)
    input_dir = input_path if input_path.is_dir() else input_path.parent
    service = CVGenerationService(output_dir=input_dir)
    validated_files = (
        service.validate_directory(input_path)
        if input_path.is_dir()
        else service.validate_files([input_path])
    )
    target = "files" if input_path.is_dir() else "file"
    typer.echo(f"Validated {len(validated_files)} CV YAML {target} in {input_path}.")


@app.command("render")
def render(
    ctx: typer.Context,
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            help="YAML CV file or directory of YAML CV files to render.",
        ),
    ],
    output_dir: Path = typer.Option(
        Path("data/cv_pdfs"),
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Directory where PDF CV files will be written.",
    ),
    template_id: Annotated[
        TemplateId | None,
        typer.Option(
            help="Optional PDF template to use instead of deterministic selection."
        ),
    ] = None,
) -> None:
    """Render one YAML CV file or a directory of YAML CV files into PDFs."""
    init_command(ctx)
    input_dir = input_path if input_path.is_dir() else input_path.parent
    service = build_pdf_rendering_service(
        input_dir=input_dir,
        output_dir=output_dir,
        template_id=template_id,
    )
    rendered_files = (
        service.render_directory(input_path)
        if input_path.is_dir()
        else service.render_files([input_path])
    )
    target = "PDFs" if input_path.is_dir() else "PDF"
    typer.echo(f"Rendered {len(rendered_files)} {target} in {output_dir}.")


@app.command("ingest")
def ingest(
    ctx: typer.Context,
    *,
    pdf_dir: Path = typer.Option(
        Path("data/cv_pdfs"),
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory of rendered CV PDFs to parse, chunk, and index.",
    ),
    reset: Annotated[
        bool,
        typer.Option("--reset", help="Recreate the Qdrant collection before indexing."),
    ] = False,
) -> IngestionSummary:
    """Parse rendered CV PDFs, chunk them, and index the chunks into Qdrant."""
    init_command(ctx)
    summary = build_cv_ingestion_service(pdf_dir=pdf_dir).ingest(reset=reset)
    typer.echo(
        f"Ingested {summary.pdf_count} PDFs into {summary.chunk_count} chunks from {pdf_dir}."
    )
    return summary


def build_pdf_rendering_service(
    *,
    input_dir: Path,
    output_dir: Path,
    template_id: TemplateId | None,
) -> "PDFRenderingService":
    """Build the PDF rendering service lazily to keep CLI help fast."""
    from cv_screener.cv_generation.pdf.renderer import (  # noqa: PLC0415
        PDFRenderingService,
    )

    return PDFRenderingService(
        input_dir=input_dir,
        output_dir=output_dir,
        template_id=template_id,
    )


def build_cv_ingestion_service(*, pdf_dir: Path) -> "CVIngestionService":
    """Build the ingestion service lazily to keep CLI import overhead low."""
    from cv_screener.ingestion import CVIngestionService  # noqa: PLC0415

    return CVIngestionService(pdf_dir=pdf_dir)


def main() -> None:
    """Run the Typer application."""
    app()
