"""CLI command registration."""

from pathlib import Path
from typing import Annotated

import typer

from cv_screener.cli.dependencies import (
    build_cv_ingestion_service,
    build_hybrid_retriever,
    build_pdf_rendering_service,
)
from cv_screener.cli.generation import generate_cv_content_files
from cv_screener.cli.runtime import init_command
from cv_screener.cv_generation.content.generator import (
    CVGenerationService,
    GenerationMode,
)
from cv_screener.cv_generation.pdf.templates import TemplateId
from cv_screener.ingestion.schema import IngestionSummary
from cv_screener.retrieval.schema import RetrievedChunk


def register_commands(app: typer.Typer) -> None:
    """Register all CLI commands on the provided Typer app."""
    app.command("generate-content")(generate_content)
    app.command("generate-cvs")(generate_cvs)
    app.command("validate")(validate)
    app.command("render")(render)
    app.command("ingest")(ingest)
    app.command("query")(query)


def generate_content(
    ctx: typer.Context,
    count: int = typer.Option(3, min=1, max=50, help="Number of YAML CVs to generate."),
    mode: Annotated[
        GenerationMode,
        typer.Option(
            help="Generation mode: seeded local samples or LLM-backed generation."
        ),
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


def generate_cvs(
    count: int = typer.Option(
        3, min=1, max=50, help="Number of CVs to generate and render."
    ),
    mode: Annotated[
        GenerationMode,
        typer.Option(
            help="Generation mode: seeded local samples or LLM-backed generation."
        ),
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
    template_id: Annotated[
        TemplateId | None,
        typer.Option(
            help="Optional PDF template to use instead of deterministic selection."
        ),
    ] = None,
) -> None:
    """Generate validated CV content YAML files and render them into PDFs."""
    written_files = generate_cv_content_files(
        ctx=None,
        count=count,
        mode=mode,
        output_dir=content_dir,
    )
    typer.echo(f"Generated {len(written_files)} CV YAML files in {content_dir}.")
    rendered_files = build_pdf_rendering_service(
        input_dir=content_dir,
        output_dir=pdf_dir,
        template_id=template_id,
    ).render_files(written_files)
    typer.echo(f"Rendered {len(rendered_files)} CV PDFs in {pdf_dir}.")


def validate(
    ctx: typer.Context,
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            help="YAML CV file or directory of YAML CV files to validate.",
        ),
    ],
) -> None:
    """Validate one YAML CV file or a directory of YAML CV files."""
    init_command(ctx)
    input_dir = input_path if input_path.is_dir() else input_path.parent
    service = CVGenerationService(output_dir=input_dir)
    validated_files = (
        service.validate_directory(input_path)
        if input_path.is_dir()
        else service.validate_files([input_path])
    )
    target = "files" if input_path.is_dir() else "file"
    typer.echo(f"Validated {len(validated_files)} CV YAML {target} in {input_path}.")


def render(
    ctx: typer.Context,
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            help="YAML CV file or directory of YAML CV files to render.",
        ),
    ],
    output_dir: Path = typer.Option(
        Path("data/cv_pdfs"),
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Directory where PDF CV files will be written.",
    ),
    template_id: Annotated[
        TemplateId | None,
        typer.Option(
            help="Optional PDF template to use instead of deterministic selection."
        ),
    ] = None,
) -> None:
    """Render one YAML CV file or a directory of YAML CV files into PDFs."""
    init_command(ctx)
    input_dir = input_path if input_path.is_dir() else input_path.parent
    service = build_pdf_rendering_service(
        input_dir=input_dir,
        output_dir=output_dir,
        template_id=template_id,
    )
    rendered_files = (
        service.render_directory(input_path)
        if input_path.is_dir()
        else service.render_files([input_path])
    )
    target = "PDFs" if input_path.is_dir() else "PDF"
    typer.echo(f"Rendered {len(rendered_files)} {target} in {output_dir}.")


def ingest(
    ctx: typer.Context,
    *,
    pdf_dir: Path = typer.Option(
        Path("data/cv_pdfs"),
        exists=True,
        file_okay=False,
        dir_okay=True,
        readable=True,
        help="Directory of rendered CV PDFs to parse, chunk, and index.",
    ),
    reset: Annotated[
        bool,
        typer.Option("--reset", help="Recreate the Qdrant collection before indexing."),
    ] = False,
) -> IngestionSummary:
    """Parse rendered CV PDFs, chunk them, and index the chunks into Qdrant."""
    init_command(ctx)
    summary = build_cv_ingestion_service(pdf_dir=pdf_dir).ingest(reset=reset)
    typer.echo(
        f"Ingested {summary.pdf_count} PDFs into {summary.chunk_count} chunks from {pdf_dir}."
    )
    return summary


def query(
    ctx: typer.Context,
    query_text: Annotated[
        str,
        typer.Argument(help="Search query to find matching CV chunks."),
    ],
    top_k: Annotated[
        int,
        typer.Option("--top-k", "-k", help="Maximum number of results to show."),
    ] = 5,
    max_length: Annotated[
        int,
        typer.Option(
            "--max-length",
            "-m",
            help="Truncate chunk text to this many characters for display.",
        ),
    ] = 200,
) -> None:
    """Search indexed CV chunks with hybrid retrieval and print results."""
    init_command(ctx)
    results = build_hybrid_retriever().retrieve(query_text)
    if not results:
        typer.echo("No matching CV chunks found.")
        return
    typer.echo(f"Top {min(top_k, len(results))} results for: {query_text}\n")
    for chunk in results[:top_k]:
        typer.echo(f"  [{chunk.rank}] (score: {chunk.score:.4f}) {_format_citation(chunk)}")
        if chunk.candidate_name:
            typer.echo(f"       Candidate: {chunk.candidate_name}")
        suffix = "..." if len(chunk.text) > max_length else ""
        typer.echo(f"       {chunk.text[:max_length]}{suffix}")
        typer.echo()


def _format_citation(chunk: RetrievedChunk) -> str:
    citation_parts = [chunk.source_file]
    if chunk.page:
        citation_parts.append(f"p.{chunk.page}")
    if chunk.section:
        citation_parts.append(chunk.section)
    return ", ".join(citation_parts)
