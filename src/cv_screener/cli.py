"""Typer CLI entrypoints for the CV screener package."""

from pathlib import Path

import typer

from cv_screener.cv_generation.generator import CVGenerationService

app = typer.Typer(help="CV screener CLI.")


@app.command("generate-cvs")
def generate_cvs(
    count: int = typer.Option(3, min=1, max=30, help="Number of YAML CVs to generate."),
    output_dir: Path = typer.Option(
        Path("data/cvs_contents"),
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Directory where YAML CVs will be written.",
    ),
) -> None:
    """Generate validated CV content YAML files."""
    service = CVGenerationService(output_dir=output_dir)
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
    service = CVGenerationService(output_dir=input_dir)
    validated_files = service.validate_directory(input_dir)
    typer.echo(f"Validated {len(validated_files)} CV YAML files in {input_dir}.")


def main() -> None:
    """Run the Typer application."""
    app()
