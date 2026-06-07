"""Shared runtime configuration for the Typer CLI."""

import os
import sys
from dataclasses import dataclass
from enum import StrEnum
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console


class LogLevel(StrEnum):
    """Supported Loguru log levels for CLI commands."""

    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


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


DEFAULT_RUNTIME_SETTINGS = CLIRuntimeSettings(
    log_level=LogLevel.INFO,
    no_progress=False,
    color=ColorMode.AUTO,
)


def configure_callback(
    ctx: typer.Context,
    *,
    log_level: Annotated[
        LogLevel,
        typer.Option(help="Global log level for CLI commands."),
    ] = LogLevel.INFO,
    no_progress: Annotated[
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
    """Configure global CLI runtime behavior."""
    ctx.obj = CLIRuntimeSettings(
        log_level=log_level,
        no_progress=no_progress,
        color=color,
    )


def init_command(ctx: typer.Context) -> tuple[CLIRuntimeSettings, Console]:
    """Initialize per-command runtime settings and logging."""
    settings = _get_runtime_settings(ctx)
    return init_runtime(settings)


def init_runtime(settings: CLIRuntimeSettings) -> tuple[CLIRuntimeSettings, Console]:
    """Initialize logging and console output for explicit runtime settings."""
    console = Console(
        file=sys.stderr,
        no_color=not should_use_color(settings.color),
        force_terminal=settings.color is ColorMode.ALWAYS,
    )
    logger.remove()
    logger.add(
        lambda message: console.print(message, end=""),
        level=settings.log_level.value.upper(),
    )
    return settings, console


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
    return (
        sys.stderr.isatty() and not is_ci_environment() and "NO_COLOR" not in os.environ
    )


def is_ci_environment() -> bool:
    """Return whether the CLI is running in a CI-like environment."""
    return os.environ.get("CI", "").strip().lower() in {"1", "true", "yes"}


def _get_runtime_settings(ctx: typer.Context) -> CLIRuntimeSettings:
    """Return the root CLI runtime settings for the current command."""
    if isinstance(ctx.obj, CLIRuntimeSettings):
        return ctx.obj
    return DEFAULT_RUNTIME_SETTINGS
