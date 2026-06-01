"""Public package entrypoints for the CV screener."""

from importlib import import_module


def main() -> None:
    """Run the CLI entrypoint lazily."""
    cli_module = import_module("cv_screener.cli")
    cli_module.main()


__all__ = ["main"]
