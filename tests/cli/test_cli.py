import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from cv_screener.cli import ColorMode, LogLevel, should_use_color, should_use_progress
from cv_screener.cli.app import app as cli_app
from cv_screener.ingestion.schema import IngestionSummary
from cv_screener.rag.service import RAGQueryResult

runner = CliRunner()
cli_app_module = importlib.import_module("cv_screener.cli.app")
cli_commands = importlib.import_module("cv_screener.cli.commands")
cli_generation = importlib.import_module("cv_screener.cli.generation")


def test_should_use_progress_requires_tty_and_non_debug(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    monkeypatch.delenv("CI", raising=False)

    assert should_use_progress(no_progress=False, log_level=LogLevel.INFO)
    assert not should_use_progress(no_progress=False, log_level=LogLevel.DEBUG)


def test_should_use_progress_disables_for_explicit_flag_or_ci(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)

    assert not should_use_progress(no_progress=True, log_level=LogLevel.INFO)
    monkeypatch.setenv("CI", "true")
    assert not should_use_progress(no_progress=False, log_level=LogLevel.INFO)


def test_should_use_color_honors_mode_and_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("sys.stderr.isatty", lambda: True)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("NO_COLOR", raising=False)

    assert should_use_color(ColorMode.ALWAYS)
    assert not should_use_color(ColorMode.NEVER)
    assert should_use_color(ColorMode.AUTO)

    monkeypatch.setenv("CI", "1")
    assert not should_use_color(ColorMode.AUTO)


def test_root_command_without_subcommand_shows_help() -> None:
    result = runner.invoke(cli_app, [])

    assert result.exit_code == 2
    assert "Usage:" in result.stdout
    assert "generate-content" in result.stdout


def test_generate_content_command_generates_yaml_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    generated_paths = [tmp_path / "candidate-1.yaml", tmp_path / "candidate-2.yaml"]
    progress_calls: list[tuple[bool, LogLevel]] = []

    class FakeGenerationService:
        def __init__(
            self, *, output_dir: Path, profile_source: object | None = None
        ) -> None:
            self.output_dir = output_dir
            self.profile_source = profile_source

        def generate(
            self, count: int, progress_callback: object | None = None
        ) -> list[Path]:
            assert self.output_dir == tmp_path
            assert count == 2
            assert progress_callback is None
            return generated_paths

    monkeypatch.setattr(cli_generation, "CVGenerationService", FakeGenerationService)
    monkeypatch.setattr(
        cli_generation,
        "should_use_progress",
        lambda **kwargs: (
            progress_calls.append((kwargs["no_progress"], kwargs["log_level"])) or False
        ),
    )

    result = runner.invoke(
        cli_app,
        [
            "--no-progress",
            "--color",
            "never",
            "generate-content",
            "--count",
            "2",
            "--output-dir",
            str(tmp_path),
        ],
    )

    assert result.exit_code == 0
    assert progress_calls == [(True, LogLevel.INFO)]
    assert result.stdout == f"Generated 2 CV YAML files in {tmp_path}.\n"


def test_generate_cvs_command_orchestrates_generation_and_rendering(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    content_dir = tmp_path / "content"
    pdf_dir = tmp_path / "pdf"
    calls: list[tuple[str, Path, Path | None]] = []
    generated_paths = [
        content_dir / "candidate-1.yaml",
        content_dir / "candidate-2.yaml",
    ]
    rendered_paths = [pdf_dir / "candidate-1.pdf", pdf_dir / "candidate-2.pdf"]
    photo_summary = SimpleNamespace(profile_count=2, generated_count=2, skipped_count=0)

    class FakeGenerationService:
        def __init__(
            self, *, output_dir: Path, profile_source: object | None = None
        ) -> None:
            del profile_source
            calls.append(("generate_init", output_dir, None))

        def generate(
            self, count: int, progress_callback: object | None = None
        ) -> list[Path]:
            assert count == 2
            assert progress_callback is None
            calls.append(("generate", content_dir, None))
            return generated_paths

    def fake_generate_photos(
        *,
        ctx: object | None,
        paths: list[Path],
        photo_dir: Path,
    ) -> object:
        assert ctx is None
        assert paths == generated_paths
        assert photo_dir == Path("data/generated/photos")
        calls.append(("photos", photo_dir, None))
        return photo_summary

    class FakePDFRenderingService:
        def __init__(
            self,
            *,
            input_dir: Path,
            output_dir: Path,
            template_id: object | None = None,
        ) -> None:
            assert template_id is None
            calls.append(("render_init", input_dir, output_dir))

        def render_files(self, paths: list[Path]) -> list[Path]:
            assert paths == generated_paths
            calls.append(("render_files", content_dir, pdf_dir))
            return rendered_paths

    monkeypatch.setattr(cli_generation, "CVGenerationService", FakeGenerationService)
    monkeypatch.setattr(
        cli_commands,
        "generate_cv_photo_files",
        fake_generate_photos,
    )
    monkeypatch.setattr(
        cli_commands,
        "build_pdf_rendering_service",
        FakePDFRenderingService,
    )
    monkeypatch.setattr(cli_generation, "should_use_progress", lambda **_: False)

    result = runner.invoke(
        cli_app,
        [
            "--no-progress",
            "--color",
            "never",
            "generate-cvs",
            "--count",
            "2",
            "--content-dir",
            str(content_dir),
            "--pdf-dir",
            str(pdf_dir),
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        ("generate_init", content_dir, None),
        ("generate", content_dir, None),
        ("photos", Path("data/generated/photos"), None),
        ("render_init", content_dir, pdf_dir),
        ("render_files", content_dir, pdf_dir),
    ]
    assert result.stdout == (
        f"Generated 2 CV YAML files in {content_dir}.\n"
        f"Prepared 2 CV photos in {Path('data/generated/photos')}.\n"
        f"Rendered 2 CV PDFs in {pdf_dir}.\n"
    )


def test_generate_photos_command_generates_for_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_dir = tmp_path / "content"
    photo_dir = tmp_path / "photos"
    input_dir.mkdir()
    summary = SimpleNamespace(profile_count=3, generated_count=2, skipped_count=1)
    calls: list[tuple[str, Path]] = []

    class FakePhotoGenerationService:
        def __init__(self, *, photo_dir: Path) -> None:
            calls.append(("init", photo_dir))

        def generate_directory(self, directory: Path) -> object:
            calls.append(("directory", directory))
            return summary

        def generate_files(self, _paths: list[Path]) -> object:
            message = "directory path should use generate_directory"
            raise AssertionError(message)

    monkeypatch.setattr(
        cli_commands,
        "build_photo_generation_service",
        FakePhotoGenerationService,
    )

    result = runner.invoke(
        cli_app,
        [
            "--color",
            "never",
            "generate-photos",
            str(input_dir),
            "--photo-dir",
            str(photo_dir),
        ],
    )

    assert result.exit_code == 0
    assert calls == [("init", photo_dir), ("directory", input_dir)]
    assert result.stdout == (f"Prepared 2 CV photos for 3 YAML files in {input_dir}.\n")


def test_render_command_renders_single_yaml_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_path = tmp_path / "candidate.yaml"
    output_dir = tmp_path / "pdf"
    rendered_path = output_dir / "candidate.pdf"
    calls: list[tuple[str, Path, Path]] = []
    input_path.write_text("candidate_id: placeholder\n", encoding="utf-8")

    class FakePDFRenderingService:
        def __init__(
            self,
            *,
            input_dir: Path,
            output_dir: Path,
            template_id: object | None = None,
        ) -> None:
            assert input_dir == input_path.parent
            assert template_id is None
            calls.append(("render_init", input_dir, output_dir))

        def render_files(self, paths: list[Path]) -> list[Path]:
            assert paths == [input_path]
            calls.append(("render_files", input_path.parent, output_dir))
            return [rendered_path]

    monkeypatch.setattr(
        cli_commands,
        "build_pdf_rendering_service",
        FakePDFRenderingService,
    )

    result = runner.invoke(
        cli_app,
        [
            "--color",
            "never",
            "render",
            str(input_path),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        ("render_init", input_path.parent, output_dir),
        ("render_files", input_path.parent, output_dir),
    ]
    assert result.stdout == f"Rendered 1 PDF in {output_dir}.\n"


def test_render_command_renders_yaml_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    input_dir = tmp_path / "content"
    output_dir = tmp_path / "pdf"
    rendered_paths = [output_dir / "candidate-1.pdf", output_dir / "candidate-2.pdf"]
    calls: list[tuple[str, Path, Path]] = []
    input_dir.mkdir()

    class FakePDFRenderingService:
        def __init__(
            self,
            *,
            input_dir: Path,
            output_dir: Path,
            template_id: object | None = None,
        ) -> None:
            assert template_id is None
            calls.append(("render_init", input_dir, output_dir))

        def render_directory(self, directory: Path | None = None) -> list[Path]:
            assert directory == input_dir
            calls.append(("render_directory", input_dir, output_dir))
            return rendered_paths

    monkeypatch.setattr(
        cli_commands,
        "build_pdf_rendering_service",
        FakePDFRenderingService,
    )

    result = runner.invoke(
        cli_app,
        [
            "--color",
            "never",
            "render",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        ("render_init", input_dir, output_dir),
        ("render_directory", input_dir, output_dir),
    ]
    assert result.stdout == f"Rendered 2 PDFs in {output_dir}.\n"


def test_ingest_command_orchestrates_pdf_ingestion(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "candidate-1.pdf").write_bytes(b"%PDF-1.4")
    (pdf_dir / "candidate-2.pdf").write_bytes(b"%PDF-1.4")
    calls: list[tuple[str, object, object | None]] = []

    class FakeIngestionService:
        def __init__(self, *, pdf_dir: Path) -> None:
            calls.append(("ingest_init", pdf_dir, None))

        def ingest(
            self,
            *,
            reset: bool = False,
            progress_callback: object | None = None,
            expected_pdf_count: int | None = None,
        ) -> object:
            _ = expected_pdf_count
            calls.append(("ingest", reset, progress_callback))
            return IngestionSummary(pdf_count=2, chunk_count=5, reset=reset)

    monkeypatch.setattr(
        cli_commands,
        "build_cv_ingestion_service",
        FakeIngestionService,
    )
    monkeypatch.setattr(cli_commands, "should_use_progress", lambda **_: False)

    result = runner.invoke(
        cli_app,
        [
            "--color",
            "never",
            "ingest",
            "--pdf-dir",
            str(pdf_dir),
            "--reset",
        ],
    )

    assert result.exit_code == 0
    assert calls == [
        ("ingest_init", pdf_dir, None),
        ("ingest", True, None),
    ]
    assert result.stdout == f"Ingested 2 PDFs into 5 chunks from {pdf_dir}.\n"


def test_ingest_command_uses_progress_callback_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    (pdf_dir / "candidate-1.pdf").write_bytes(b"%PDF-1.4")
    (pdf_dir / "candidate-2.pdf").write_bytes(b"%PDF-1.4")
    captured: dict[str, object | None] = {"callback": None}

    class FakeIngestionService:
        def __init__(self, *, pdf_dir: Path) -> None:
            _ = pdf_dir

        def ingest(
            self,
            *,
            reset: bool = False,
            progress_callback: object | None = None,
            expected_pdf_count: int | None = None,
        ) -> object:
            _ = (reset, expected_pdf_count)
            captured["callback"] = progress_callback
            return IngestionSummary(pdf_count=2, chunk_count=5, reset=False)

    monkeypatch.setattr(
        cli_commands,
        "build_cv_ingestion_service",
        FakeIngestionService,
    )
    monkeypatch.setattr(cli_commands, "should_use_progress", lambda **_: True)

    result = runner.invoke(
        cli_app,
        [
            "--color",
            "never",
            "ingest",
            "--pdf-dir",
            str(pdf_dir),
        ],
    )

    assert result.exit_code == 0
    assert callable(captured["callback"])


def test_query_command_runs_rag_service_and_prints_grounded_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    class FakeRAGQueryService:
        def run(self, query_text: str) -> RAGQueryResult:
            calls.append(query_text)
            return RAGQueryResult(
                final_text=(
                    "Ada Lovelace has Python backend experience.\n\n"
                    "Sources:\n"
                    "- ada-lovelace.pdf (page 2, Experience)"
                ),
                state={"final_text": "unused"},
            )

    monkeypatch.setattr(
        cli_commands,
        "build_rag_query_service",
        lambda **_: FakeRAGQueryService(),
    )

    result = runner.invoke(
        cli_app,
        [
            "--color",
            "never",
            "query",
            "Who has Python backend experience?",
        ],
    )

    assert result.exit_code == 0
    assert calls == ["Who has Python backend experience?"]
    assert result.stdout == (
        "Ada Lovelace has Python backend experience.\n\n"
        "Sources:\n"
        "- ada-lovelace.pdf (page 2, Experience)\n"
    )


def test_serve_command_runs_chainlit_launcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[cli_commands.ChainlitServeRequest] = []

    class FakeLauncher:
        def run(self, request: cli_commands.ChainlitServeRequest) -> int:
            calls.append(request)
            return 0

    monkeypatch.setattr(cli_commands, "build_chainlit_launcher", FakeLauncher)

    result = runner.invoke(
        cli_app,
        [
            "--color",
            "never",
            "serve",
            "--host",
            "127.0.0.1",
            "--port",
            "9000",
            "--open-browser",
            "--watch",
        ],
    )

    assert result.exit_code == 0
    assert len(calls) == 1
    request = calls[0]
    assert request.host == "127.0.0.1"
    assert request.port == 9000
    assert request.headless is False
    assert request.watch is True
