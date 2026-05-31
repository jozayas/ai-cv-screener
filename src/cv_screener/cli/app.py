"""Typer application shell."""

from typing import Annotated

import typer

from cv_screener.cli.commands import register_commands
from cv_screener.cli.runtime import ColorMode, LogLevel, configure_callback

app = typer.Typer(
    help="CV screener CLI.",
    no_args_is_help=True,
)


@app.callback()
def main_callback(
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
    """CV screener CLI."""
    configure_callback(
        ctx=ctx,
        log_level=log_level,
        no_progress=no_progress,
        color=color,
    )


register_commands(app)


def main() -> None:
    """Run the Typer application."""
    app()
