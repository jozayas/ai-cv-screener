"""Helpers for CV content generation CLI commands."""

from pathlib import Path

import typer
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from cv_screener.cli.runtime import init_command, should_use_progress
from cv_screener.cv_generation.content.generator import (
    CVGenerationService,
    CVProfileSource,
    GenerationMode,
)


def generate_cv_content_files(
    *,
    ctx: typer.Context,
    count: int,
    mode: GenerationMode,
    output_dir: Path,
) -> list[Path]:
    """Generate YAML CV content files with command-configured runtime behavior."""
    runtime, console = init_command(ctx)
    service = CVGenerationService(
        output_dir=output_dir,
        profile_source=build_profile_source(mode),
    )
    if not should_use_progress(
        no_progress=runtime.no_progress,
        log_level=runtime.log_level,
    ):
        return service.generate(count=count)

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


def build_profile_source(mode: GenerationMode) -> CVProfileSource:
    """Build the profile source for the selected generation mode."""
    from cv_screener.cv_generation.content.sources import (  # noqa: PLC0415
        OpenAICVProfileSource,
        SeededCVProfileSource,
    )

    if mode is GenerationMode.LLM:
        from cv_screener.config import GenerationSettings  # noqa: PLC0415

        return OpenAICVProfileSource(GenerationSettings())
    return SeededCVProfileSource()
