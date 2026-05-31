"""CLI command registration."""

from pathlib import Path
from typing import Annotated

import typer
from rich.progress import BarColumn, Progress, TaskProgressColumn, TextColumn

from cv_screener.cli.dependencies import (
    build_chainlit_launcher,
    build_cv_ingestion_service,
    build_pdf_rendering_service,
    build_photo_generation_service,
    build_rag_query_service,
)
from cv_screener.cli.generation import (
    generate_cv_content_files,
    generate_cv_photo_files,
)
from cv_screener.cli.runtime import init_command, should_use_progress
from cv_screener.cli.serve import ChainlitServeRequest, default_chainlit_app_path
from cv_screener.config import LookupSettings
from cv_screener.cv_generation.content.generator import (
    CVGenerationService,
    GenerationMode,
)
from cv_screener.cv_generation.pdf.templates import TemplateId
from cv_screener.cv_generation.photos.service import PhotoGenerationSummary
from cv_screener.ingestion.schema import IngestionSummary


def register_commands(app: typer.Typer) -> None:
    """Register all CLI commands on the provided Typer app."""
    app.command("generate-content")(generate_content)
    app.command("generate-cvs")(generate_cvs)
    app.command("generate-photos")(generate_photos)
    app.command("validate")(validate)
    app.command("render")(render)
    app.command("ingest")(ingest)
    app.command("query")(query)
    app.command("serve")(serve)


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
    photo_dir: Path = typer.Option(
        Path("data/generated/photos"),
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Directory where generated CV photos will be written.",
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
    photo_summary = generate_cv_photo_files(
        ctx=None,
        paths=written_files,
        photo_dir=photo_dir,
    )
    typer.echo(f"Prepared {photo_summary.generated_count} CV photos in {photo_dir}.")
    rendered_files = build_pdf_rendering_service(
        input_dir=content_dir,
        output_dir=pdf_dir,
        template_id=template_id,
    ).render_files(written_files)
    typer.echo(f"Rendered {len(rendered_files)} CV PDFs in {pdf_dir}.")


def generate_photos(
    ctx: typer.Context,
    input_path: Annotated[
        Path,
        typer.Argument(
            exists=True,
            file_okay=True,
            dir_okay=True,
            readable=True,
            help="YAML CV file or directory of YAML CV files to update with photos.",
        ),
    ],
    photo_dir: Path = typer.Option(
        Path("data/generated/photos"),
        file_okay=False,
        dir_okay=True,
        writable=True,
        help="Directory where generated CV photos will be written.",
    ),
) -> PhotoGenerationSummary:
    """Generate synthetic headshots for YAML CV profiles."""
    init_command(ctx)
    service = build_photo_generation_service(photo_dir=photo_dir)
    summary = (
        service.generate_directory(input_path)
        if input_path.is_dir()
        else service.generate_files([input_path])
    )
    target = "files" if input_path.is_dir() else "file"
    typer.echo(
        f"Prepared {summary.generated_count} CV photos for {summary.profile_count} YAML {target} in {input_path}."
    )
    return summary


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
    runtime, console = init_command(ctx)
    service = build_cv_ingestion_service(pdf_dir=pdf_dir)
    pdf_count = len(sorted(pdf_dir.glob("*.pdf")))

    if (
        not should_use_progress(
            no_progress=runtime.no_progress,
            log_level=runtime.log_level,
        )
        or pdf_count == 0
    ):
        summary = service.ingest(reset=reset)
    else:
        total_steps = pdf_count + 3
        console.print()
        with Progress(
            TextColumn("{task.fields[stage]}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("({task.completed:.0f}/{task.total:.0f})"),
            console=console,
        ) as progress_bar:
            task_id = progress_bar.add_task(
                "ingest",
                total=total_steps,
                stage="Ingesting PDFs",
            )

            def on_progress(current: int, total: int, stage: str) -> None:
                target_total = total or total_steps
                if target_total != total_steps:
                    return
                progress_bar.update(
                    task_id,
                    completed=max(0, min(current, total_steps)),
                    stage=stage,
                )

            summary = service.ingest(
                reset=reset,
                expected_pdf_count=pdf_count,
                progress_callback=on_progress,
            )
    typer.echo(
        f"Ingested {summary.pdf_count} PDFs into {summary.chunk_count} chunks from {pdf_dir}."
    )
    return summary


def query(
    ctx: typer.Context,
    query_text: Annotated[
        str,
        typer.Argument(help="Recruiter-style question to answer from indexed CVs."),
    ],
) -> None:
    """Answer a recruiter-style question from indexed CV content."""
    init_command(ctx)
    result = build_rag_query_service(
        candidate_name_min_score=LookupSettings().candidate_name_min_score,
    ).run(query_text)
    typer.echo(result.final_text)


def serve(
    ctx: typer.Context,
    *,
    host: str = typer.Option("127.0.0.1", help="Host interface for the Chainlit UI."),
    port: int = typer.Option(8000, min=1, max=65535, help="Port for the Chainlit UI."),
    headless: Annotated[
        bool,
        typer.Option(
            "--headless/--open-browser",
            help="Run without asking Chainlit to open a browser window.",
        ),
    ] = True,
    watch: Annotated[
        bool,
        typer.Option(
            "--watch/--no-watch",
            help="Reload the app when source files change.",
        ),
    ] = False,
) -> None:
    """Serve the Chainlit chat UI on top of the existing RAG runtime."""
    init_command(ctx)
    request = ChainlitServeRequest(
        app_path=default_chainlit_app_path(),
        host=host,
        port=port,
        headless=headless,
        watch=watch,
    )
    exit_code = build_chainlit_launcher().run(request)
    if exit_code != 0:
        raise typer.Exit(code=exit_code)
