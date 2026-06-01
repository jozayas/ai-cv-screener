from pathlib import Path

import pytest
from pydantic import ValidationError

from cv_screener.cv_generation.content import generator as generator_module
from cv_screener.cv_generation.content.generator import CVGenerationService
from cv_screener.cv_generation.content.schema import CVProfileDraft
from cv_screener.cv_generation.content.yaml_io import load_cv_profile


def test_generated_yaml_profiles_validate_against_schema(tmp_path: Path) -> None:
    service = CVGenerationService(output_dir=tmp_path)
    generated = service.generate(count=3)

    assert len(generated) == 3
    for path in generated:
        profile = load_cv_profile(path)
        assert profile.candidate_id
        assert profile.full_name


def test_validate_directory_accepts_generated_profiles(tmp_path: Path) -> None:
    service = CVGenerationService(output_dir=tmp_path)
    service.generate(count=2)

    validated = service.validate_directory(tmp_path)

    assert len(validated) == 2
    assert all(path.exists() for path in validated)


def test_candidate_profile_rejects_missing_required_fields(tmp_path: Path) -> None:
    invalid_yaml = tmp_path / "invalid.yaml"
    invalid_yaml.write_text("full_name: Test User\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        load_cv_profile(invalid_yaml)


def test_generation_respects_profile_source_concurrency_cap(tmp_path: Path) -> None:
    class LowConcurrencySource:
        @property
        def max_concurrency(self) -> int:
            return 1

        def generate_draft(self, *, index: int) -> CVProfileDraft:
            return CVProfileDraft.model_validate(
                {
                    "full_name": f"Candidate {index}",
                    "email": f"candidate{index}@example.com",
                    "phone": "+34123456789",
                    "location": "Barcelona",
                    "professional_summary": "Backend engineer.",
                    "skills": ["Python"],
                    "experience": [
                        {
                            "company": "Leadtech",
                            "role": "Engineer",
                            "start_date": "2022-01-01",
                            "end_date": None,
                            "summary": "Built APIs.",
                            "highlights": [],
                            "technologies": [],
                        }
                    ],
                    "education": [
                        {
                            "institution": "UPC",
                            "degree": "BSc",
                            "field_of_study": "Computer Science",
                            "graduation_year": 2021,
                        }
                    ],
                }
            )

    service = CVGenerationService(
        output_dir=tmp_path,
        profile_source=LowConcurrencySource(),
    )

    captured_workers: list[int] = []
    original_executor = generator_module.ThreadPoolExecutor

    class CapturingExecutor(original_executor):
        def __init__(self, max_workers: int | None = None) -> None:
            if max_workers is not None:
                captured_workers.append(max_workers)
            super().__init__(max_workers=max_workers)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(generator_module, "ThreadPoolExecutor", CapturingExecutor)
    try:
        _ = service.generate(count=3)
    finally:
        monkeypatch.undo()

    assert captured_workers == [1]
