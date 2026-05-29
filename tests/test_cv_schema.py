from pathlib import Path
from uuid import UUID

import pytest
from pydantic import ValidationError

from cv_screener.cv_generation.generator import CVGenerationService
from cv_screener.cv_generation.schema import CVProfile
from cv_screener.cv_generation.yaml_io import load_cv_profile


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
