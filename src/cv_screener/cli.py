"""Typer CLI entrypoints for the CV screener package."""

import os
import sys
from dataclasses import dataclass
from enum import StrEnum
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
)
from cv_screener.cv_generation.content.sources import (
    OpenAICVProfileSource,
    SeededCVProfileSource,
)
from cv_screener.cv_generation.pdf import PDFRenderingService

app = typer.Typer(help="CV screener CLI.")


class LogLevel(StrEnum):
    """Supported Loguru log levels for CLI commands."""

    TRACE = "TRACE"
    DEBUG = "DEBUG"
    INFO = "INFO"
    SUCCESS = "SUCCESS"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class ColorMode(StrEnum):
    """Supported color modes for CLI output."""

    AUTO = "auto"
    ALWAYS = "always"
    NEVER = "never"


@dataclass(frozen=True)
class CLIRuntimeSettings:
    """Shared CLI runtime settings configured at the app level."""

    log_level: LogLevel
    no_progress: bool
    color: ColorMode


@app.callback()
def main_callback(
    ctx: typer.Context,
    log_level: Annotated[
        str,
        typer.Option(help="Global log level for CLI commands."),
    ] = "INFO",
    no_progress: Annotated[
        bool | None,
        typer.Option(
            "--no-progress",
            help="Disable progress bars and similar interactive status output.",
        ),
    ] = None,
    color: Annotated[
        ColorMode,
        typer.Option(help="Color mode for CLI output."),
    ] = ColorMode.AUTO,
) -> None:
    """Configure global CLI runtime behavior."""
    ctx.obj = CLIRuntimeSettings(
        log_level=normalize_log_level(log_level),
        no_progress=bool(no_progress),
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
    settings, console = init_command(ctx)
    written_files = run_cv_content_generation(
        count=count,
        mode=mode,
        output_dir=output_dir,
        log_level=settings.log_level,
        no_progress=settings.no_progress,
        console=console,
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
    settings, console = init_command(ctx)
    written_files = run_cv_content_generation(
        count=count,
        mode=mode,
        output_dir=content_dir,
        log_level=settings.log_level,
        no_progress=settings.no_progress,
        console=console,
    )
    typer.echo(f"Generated {len(written_files)} CV YAML files in {content_dir}.")
    rendered_files = run_pdf_rendering(input_dir=content_dir, output_dir=pdf_dir)
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
    service = CVGenerationService(output_dir=input_dir)
    validated_files = service.validate_directory(input_dir)
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
    rendered_files = run_pdf_rendering(input_dir=input_dir, output_dir=output_dir)
    typer.echo(f"Rendered {len(rendered_files)} CV PDFs in {output_dir}.")


def run_cv_content_generation(  # noqa: PLR0913
    *,
    count: int,
    mode: GenerationMode,
    output_dir: Path,
    log_level: LogLevel,
    no_progress: bool,
    console: Console,
) -> list[Path]:
    """Run YAML CV generation with the configured profile source and progress output."""
    profile_source = build_profile_source(mode)
    service = CVGenerationService(output_dir=output_dir, profile_source=profile_source)
    progress_enabled = should_use_progress(no_progress=no_progress, log_level=log_level)
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
            return service.generate(
                count=count,
                progress_callback=lambda _current, _total: progress_bar.advance(task_id, 1),
            )
    return service.generate(count=count)


def run_pdf_rendering(*, input_dir: Path, output_dir: Path) -> list[Path]:
    """Render PDFs from validated YAML CV content files."""
    service = PDFRenderingService(input_dir=input_dir, output_dir=output_dir)
    return service.render_directory()


def build_profile_source(mode: GenerationMode) -> SeededCVProfileSource | OpenAICVProfileSource:
    """Build the profile source for the selected generation mode."""
    if mode is GenerationMode.LLM:
        return OpenAICVProfileSource(GenerationSettings())
    return SeededCVProfileSource()


def get_runtime_settings(ctx: typer.Context) -> CLIRuntimeSettings:
    """Return the root CLI runtime settings for the current command."""
    return ctx.obj if isinstance(ctx.obj, CLIRuntimeSettings) else CLIRuntimeSettings(
        log_level=LogLevel.INFO,
        no_progress=False,
        color=ColorMode.AUTO,
    )


def init_command(ctx: typer.Context) -> tuple[CLIRuntimeSettings, Console]:
    """Initialize per-command runtime settings and logging."""
    settings = get_runtime_settings(ctx)
    console = build_console(color=settings.color)
    configure_logging(console=console, log_level=settings.log_level)
    return settings, console


def build_console(*, color: ColorMode) -> Console:
    """Create the stderr-backed console used by the CLI."""
    color_enabled = should_use_color(color)
    return Console(
        file=sys.stderr,
        no_color=not color_enabled,
        force_terminal=color is ColorMode.ALWAYS,
    )


def configure_logging(*, console: Console, log_level: LogLevel) -> None:
    """Configure Loguru output for CLI commands."""
    logger.remove()
    logger.add(
        lambda message: console.print(message, end=""),
        level=log_level,
    )


def normalize_log_level(value: str) -> LogLevel:
    """Validate and normalize a user-provided Loguru log level name."""
    level_name = value.upper()
    try:
        return LogLevel(level_name)
    except ValueError as error:
        valid_levels = ", ".join(level.value for level in LogLevel)
        message = f"Unsupported log level '{value}'. Choose one of: {valid_levels}."
        raise typer.BadParameter(message) from error


def should_use_progress(*, no_progress: bool, log_level: LogLevel) -> bool:
    """Resolve whether the CLI should render a progress bar."""
    if no_progress or is_ci_environment():
        return False
    return sys.stderr.isatty() and log_level not in {LogLevel.TRACE, LogLevel.DEBUG}


def should_use_color(color: ColorMode) -> bool:
    """Resolve whether CLI output should use color and Rich terminal formatting."""
    if color is ColorMode.ALWAYS:
        return True
    if color is ColorMode.NEVER:
        return False
    return sys.stderr.isatty() and not is_ci_environment() and "NO_COLOR" not in os.environ


def is_ci_environment() -> bool:
    """Return whether the CLI is running in a CI-like environment."""
    return os.environ.get("CI", "").strip().lower() in {"1", "true", "yes"}


def main() -> None:
    """Run the Typer application."""
    app()
