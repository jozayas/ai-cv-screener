"""Typer application and command definitions."""

from pathlib import Path
from typing import Annotated

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
from cv_screener.cv_generation.pdf import PDFRenderingService

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


@app.command("generate-cv-content")
def generate_cv_content(
    ctx: typer.Context,
    count: int = typer.Option(3, min=1, max=50, help="Number of YAML CVs to generate."),
    mode: Annotated[
        GenerationMode,
        typer.Option(help="Generation mode: seeded local samples or LLM-backed generation."),
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
def generate_cvs(
    ctx: typer.Context,
    count: int = typer.Option(3, min=1, max=50, help="Number of CVs to generate and render."),
    mode: Annotated[
        GenerationMode,
        typer.Option(help="Generation mode: seeded local samples or LLM-backed generation."),
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
) -> None:
    """Generate validated CV content YAML files and render them into PDFs."""
    written_files = generate_cv_content_files(
        ctx=ctx,
        count=count,
        mode=mode,
        output_dir=content_dir,
    )
    typer.echo(f"Generated {len(written_files)} CV YAML files in {content_dir}.")
    rendered_files = PDFRenderingService(
        input_dir=content_dir,
        output_dir=pdf_dir,
    ).render_directory()
    typer.echo(f"Rendered {len(rendered_files)} CV PDFs in {pdf_dir}.")


@app.command("validate-cvs")
def validate_cvs(
    ctx: typer.Context,
    input_dir: Path = typer.Option(
        Path("data/cvs_contents"),
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory containing YAML CV files.",
    ),
) -> None:
    """Validate generated CV content YAML files."""
    init_command(ctx)
    validated_files = CVGenerationService(output_dir=input_dir).validate_directory(input_dir)
    typer.echo(f"Validated {len(validated_files)} CV YAML files in {input_dir}.")


@app.command("render-pdfs")
def render_pdfs(
    ctx: typer.Context,
    input_dir: Path = typer.Option(
        Path("data/cvs_contents"),
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory containing YAML CV files.",
    ),
    output_dir: Path = typer.Option(
        Path("data/cv_pdfs"),
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Directory where PDF CV files will be written.",
    ),
) -> None:
    """Render generated YAML CV files into PDFs."""
    init_command(ctx)
    rendered_files = PDFRenderingService(
        input_dir=input_dir,
        output_dir=output_dir,
    ).render_directory()
    typer.echo(f"Rendered {len(rendered_files)} CV PDFs in {output_dir}.")


def main() -> None:
    """Run the Typer application."""
    app()
