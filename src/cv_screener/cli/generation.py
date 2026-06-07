"""Helpers for CV content generation CLI commands."""

from pathlib import Path

from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from cv_screener.cli.dependencies import build_photo_generation_service
from cv_screener.cli.runtime import (
    CLIRuntimeSettings,
    init_runtime,
    should_use_progress,
)
from cv_screener.config import GenerationConfig
from cv_screener.cv_generation.content.generator import (
    CVGenerationService,
    CVProfileSource,
    GenerationMode,
)
from cv_screener.cv_generation.content.sources import (
    OpenAICVProfileSource,
    SeededCVProfileSource,
)
from cv_screener.cv_generation.photos.service import PhotoGenerationSummary


def generate_cv_content_files(
    *,
    runtime: CLIRuntimeSettings,
    count: int,
    mode: GenerationMode,
    output_dir: Path,
    generation_settings: GenerationConfig,
) -> list[Path]:
    """Generate YAML CV content files with command-configured runtime behavior."""
    runtime, console = init_runtime(runtime)
    service = CVGenerationService(
        output_dir=output_dir,
        profile_source=build_profile_source(
            mode,
            generation_settings=generation_settings,
        ),
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


def build_profile_source(
    mode: GenerationMode,
    *,
    generation_settings: GenerationConfig,
) -> CVProfileSource:
    """Build the profile source for the selected generation mode."""
    if mode is GenerationMode.LLM:
        return OpenAICVProfileSource(generation_settings)

    return SeededCVProfileSource()


def generate_cv_photo_files(
    *,
    paths: list[Path],
    photo_dir: Path,
    generation_settings: GenerationConfig,
) -> PhotoGenerationSummary:
    """Generate or refresh CV photos for a set of YAML profiles."""
    service = build_photo_generation_service(
        photo_dir=photo_dir,
        generation_settings=generation_settings,
    )
    return service.generate_files(paths)
