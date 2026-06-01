"""Helpers for CV content generation CLI commands."""

from importlib import import_module
from pathlib import Path
from typing import cast

import typer
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from cv_screener.cli.dependencies import build_photo_generation_service
from cv_screener.cli.runtime import init_command, should_use_progress
from cv_screener.cv_generation.content.generator import (
    CVGenerationService,
    CVProfileSource,
    GenerationMode,
)
from cv_screener.cv_generation.photos.service import PhotoGenerationSummary


def generate_cv_content_files(
    *,
    ctx: typer.Context | None,
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
    sources_module = import_module("cv_screener.cv_generation.content.sources")

    if mode is GenerationMode.LLM:
        config_module = import_module("cv_screener.config")
        settings_type = cast("type[object]", config_module.GenerationSettings)
        source = sources_module.OpenAICVProfileSource(settings_type())
        return cast("CVProfileSource", source)

    source = sources_module.SeededCVProfileSource()
    return cast("CVProfileSource", source)


def generate_cv_photo_files(
    *,
    ctx: typer.Context | None,
    paths: list[Path],
    photo_dir: Path,
) -> PhotoGenerationSummary:
    """Generate or refresh CV photos for a set of YAML profiles."""
    _ = ctx
    service = build_photo_generation_service(photo_dir=photo_dir)
    return service.generate_files(paths)
