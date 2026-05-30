import importlib
from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError
from typer.testing import CliRunner

from cv_screener.cli import ColorMode, LogLevel, should_use_color, should_use_progress
from cv_screener.cli.app import app as cli_app
from cv_screener.cv_generation.content.generator import (
    CVGenerationService,
)
from cv_screener.cv_generation.content.llm.parsing import parse_json_payload
from cv_screener.cv_generation.content.schema import CVProfile, CVProfileDraft
from cv_screener.cv_generation.content.yaml_io import load_cv_profile

runner = CliRunner()
cli_app_module = importlib.import_module("cv_screener.cli.app")
cli_generation = importlib.import_module("cv_screener.cli.generation")


def test_cv_profile_requires_valid_candidate_id() -> None:
    with pytest.raises(ValidationError):
        CVProfile.model_validate(
            {
                "candidate_id": "not-a-uuid",
                "full_name": "Test Candidate",
                "email": "test@example.com",
                "phone": "+34 600 000 000",
                "location": "Madrid, Spain",
                "professional_summary": "Summary",
                "skills": ["Python"],
                "experience": [
                    {
                        "company": "Example",
                        "role": "Engineer",
                        "start_date": "2020-01-01",
                        "end_date": None,
                        "summary": "Summary",
                        "highlights": [],
                        "technologies": ["Python"],
                    }
                ],
                "education": [
                    {
                        "institution": "Example University",
                        "degree": "BSc",
                        "field_of_study": "Computer Science",
                        "graduation_year": 2020,
                    }
                ],
            }
        )


def test_generate_writes_valid_yaml_files(tmp_path: Path) -> None:
    service = CVGenerationService(output_dir=tmp_path)

    written_files = service.generate(count=3)

    loaded = [load_cv_profile(path) for path in written_files]
    assert len(written_files) == 3
    assert len({path.name for path in written_files}) == 3
    assert all(path.suffix == ".yaml" for path in written_files)
    assert all(path.stem.endswith(str(profile.candidate_id)) for path, profile in zip(written_files, loaded, strict=True))
    assert all(path.stem.startswith("marta-alvarez") or path.stem.startswith("daniel-romero") or path.stem.startswith("lucia-moreno") for path in written_files)
    assert all(isinstance(profile.candidate_id, UUID) for profile in loaded)


def test_generate_produces_unique_candidate_ids_across_runs(tmp_path: Path) -> None:
    service = CVGenerationService(output_dir=tmp_path)
    first_batch = service.generate(count=2)

    second_batch = service.generate(count=2)

    all_names = [path.name for path in [*first_batch, *second_batch]]
    assert len(all_names) == len(set(all_names))


class FakeLLMSource:
    def generate_draft(self, *, index: int) -> CVProfileDraft:
        return CVProfileDraft.model_validate(
            {
                "full_name": f"Candidate {index}",
                "email": f"candidate{index}@example.com",
                "phone": "+34 600 000 000",
                "location": "Madrid, Spain",
                "professional_summary": "Synthetic model-generated candidate.",
                "skills": ["Python", "Docker"],
                "experience": [
                    {
                        "company": "Example Corp",
                        "role": "Backend Engineer",
                        "start_date": "2022-01-01",
                        "end_date": None,
                        "summary": "Built internal APIs.",
                        "highlights": ["Improved service reliability."],
                        "technologies": ["Python", "FastAPI"],
                    }
                ],
                "education": [
                    {
                        "institution": "Example University",
                        "degree": "BSc",
                        "field_of_study": "Computer Science",
                        "graduation_year": 2021,
                    }
                ],
            }
        )


def test_generate_supports_llm_mode_with_injected_source(tmp_path: Path) -> None:
    service = CVGenerationService(output_dir=tmp_path, profile_source=FakeLLMSource())

    written_files = service.generate(count=2)

    loaded = [load_cv_profile(path) for path in written_files]
    assert len(loaded) == 2
    assert all(profile.full_name.startswith("Candidate ") for profile in loaded)


def test_parse_json_payload_tolerates_fences_and_control_characters() -> None:
    raw_content = """```json
{
  "full_name": "Test Candidate",
  "email": "test@example.com",
  "phone": "+34 600 000 000",
  "location": "Madrid, Spain",
  "professional_summary": "Line one
Line two",
  "skills": ["Python"],
  "experience": [{
    "company": "Example",
    "role": "Engineer",
    "start_date": "2020-01-01",
    "end_date": null,
    "summary": "Built APIs.",
    "highlights": ["Did a thing."],
    "technologies": ["Python"]
  }],
  "education": [{
    "institution": "Example University",
    "degree": "BSc",
    "field_of_study": "Computer Science",
    "graduation_year": 2020
  }]
}
```"""

    payload = parse_json_payload(raw_content)

    assert payload["full_name"] == "Test Candidate"
    assert payload["professional_summary"] == "Line one Line two"


def test_experience_dates_are_validated() -> None:
    with pytest.raises(ValidationError):
        CVProfile.model_validate(
            {
                "candidate_id": "04d36682-1301-4e51-b4df-49bac4f74f7b",
                "full_name": "Test Candidate",
                "email": "test@example.com",
                "phone": "+34 600 000 000",
                "location": "Madrid, Spain",
                "professional_summary": "Summary",
                "skills": ["Python"],
                "experience": [
                    {
                        "company": "Example",
                        "role": "Engineer",
                        "start_date": "2021-01-01",
                        "end_date": "2020-01-01",
                        "summary": "Summary",
                        "highlights": [],
                        "technologies": ["Python"],
                    }
                ],
                "education": [
                    {
                        "institution": "Example University",
                        "degree": "BSc",
                        "field_of_study": "Computer Science",
                        "graduation_year": 2020,
                    }
                ],
            }
        )


def test_validate_directory_loads_generated_yaml(tmp_path: Path) -> None:
    service = CVGenerationService(output_dir=tmp_path)
    service.generate(count=2)

    validated_files = service.validate_directory(tmp_path)

    assert len(validated_files) == 2


def test_should_use_progress_requires_tty_and_non_debug(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_should_use_color_honors_mode_and_environment(monkeypatch: pytest.MonkeyPatch) -> None:
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
    assert "generate-cv-content" in result.stdout


def test_generate_cv_content_command_generates_yaml_only(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    generated_paths = [tmp_path / "candidate-1.yaml", tmp_path / "candidate-2.yaml"]
    progress_calls: list[tuple[bool, LogLevel]] = []

    class FakeGenerationService:
        def __init__(self, *, output_dir: Path, profile_source: object | None = None) -> None:
            self.output_dir = output_dir
            self.profile_source = profile_source

        def generate(self, count: int, progress_callback: object | None = None) -> list[Path]:
            assert self.output_dir == tmp_path
            assert count == 2
            assert progress_callback is None
            return generated_paths

    monkeypatch.setattr(cli_generation, "CVGenerationService", FakeGenerationService)
    monkeypatch.setattr(
        cli_generation,
        "should_use_progress",
        lambda **kwargs: progress_calls.append(
            (kwargs["no_progress"], kwargs["log_level"])
        )
        or False,
    )

    result = runner.invoke(
        cli_app,
        [
            "--no-progress",
            "--color",
            "never",
            "generate-cv-content",
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
    generated_paths = [content_dir / "candidate-1.yaml", content_dir / "candidate-2.yaml"]
    rendered_paths = [pdf_dir / "candidate-1.pdf", pdf_dir / "candidate-2.pdf"]

    class FakeGenerationService:
        def __init__(self, *, output_dir: Path, profile_source: object | None = None) -> None:
            del profile_source
            calls.append(("generate_init", output_dir, None))

        def generate(self, count: int, progress_callback: object | None = None) -> list[Path]:
            assert count == 2
            assert progress_callback is None
            calls.append(("generate", content_dir, None))
            return generated_paths

    class FakePDFRenderingService:
        def __init__(self, *, input_dir: Path, output_dir: Path) -> None:
            calls.append(("render_init", input_dir, output_dir))

        def render_directory(self) -> list[Path]:
            calls.append(("render", content_dir, pdf_dir))
            return rendered_paths

    monkeypatch.setattr(cli_generation, "CVGenerationService", FakeGenerationService)
    monkeypatch.setattr(cli_app_module, "PDFRenderingService", FakePDFRenderingService)
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
        ("render_init", content_dir, pdf_dir),
        ("render", content_dir, pdf_dir),
    ]
    assert result.stdout == f"Generated 2 CV YAML files in {content_dir}.\nRendered 2 CV PDFs in {pdf_dir}.\n"
