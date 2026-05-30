"""Typer CLI entrypoints for the CV screener package."""

import sys
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from cv_screener.config import GenerationSettings
from cv_screener.cv_generation.content.generator import (
    CVGenerationService,
    GenerationMode,
    LogVerbosity,
    ProgressMode,
)
from cv_screener.cv_generation.content.sources import (
    OpenAICVProfileSource,
    SeededCVProfileSource,
)
from cv_screener.cv_generation.pdf import PDFRenderingService

app = typer.Typer(help="CV screener CLI.")


@app.command("generate-cvs")
def generate_cvs(
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
    log_level: Annotated[
        LogVerbosity,
        typer.Option(help="Log level during generation."),
    ] = LogVerbosity.INFO,
    progress: Annotated[
        ProgressMode,
        typer.Option(help="Progress bar mode."),
    ] = ProgressMode.AUTO,
) -> None:
    """Generate validated CV content YAML files."""
    console = build_console()
    configure_logging(console=console, log_level=log_level)
    profile_source = SeededCVProfileSource()
    if mode is GenerationMode.LLM:
        profile_source = OpenAICVProfileSource(GenerationSettings())
    service = CVGenerationService(output_dir=output_dir, profile_source=profile_source)
    progress_enabled = should_use_progress(progress=progress, log_level=log_level)
    if progress_enabled:
        console.print()
        with Progress(
            TextColumn("Generating CVs"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("({task.completed:.0f}/{task.total:.0f} generated)"),
            console=console,
        ) as progress_bar:
            task_id = progress_bar.add_task("generate-cvs", total=count)
            written_files = service.generate(
                count=count,
                progress_callback=lambda _current, _total: progress_bar.advance(task_id, 1),
            )
    else:
        written_files = service.generate(count=count)
    typer.echo(f"Generated {len(written_files)} CV YAML files in {output_dir}.")


@app.command("validate-cvs")
def validate_cvs(
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
    configure_logging(console=build_console(), log_level=LogVerbosity.INFO)
    service = CVGenerationService(output_dir=input_dir)
    validated_files = service.validate_directory(input_dir)
    typer.echo(f"Validated {len(validated_files)} CV YAML files in {input_dir}.")


@app.command("render-pdfs")
def render_pdfs(
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
    configure_logging(console=build_console(), log_level=LogVerbosity.INFO)
    service = PDFRenderingService(input_dir=input_dir, output_dir=output_dir)
    rendered_files = service.render_directory()
    typer.echo(f"Rendered {len(rendered_files)} CV PDFs in {output_dir}.")


def build_console() -> Console:
    """Create the stderr-backed console used by the CLI."""
    return Console(file=sys.stderr)


def configure_logging(*, console: Console, log_level: LogVerbosity) -> None:
    """Configure Loguru output for CLI commands."""
    logger.remove()
    logger.add(
        lambda message: console.print(message, end=""),
        level=log_level.value.upper(),
    )


def should_use_progress(*, progress: ProgressMode, log_level: LogVerbosity) -> bool:
    """Resolve whether the CLI should render a progress bar."""
    if progress is ProgressMode.ON:
        return log_level not in {LogVerbosity.TRACE, LogVerbosity.DEBUG}
    if progress is ProgressMode.OFF:
        return False
    return sys.stderr.isatty() and log_level not in {LogVerbosity.TRACE, LogVerbosity.DEBUG}


def main() -> None:
    """Run the Typer application."""
    app()
