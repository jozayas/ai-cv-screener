from pathlib import Path

import pytest
from pydantic import ValidationError

from cv_screener.cv_generation.content.generator import CVGenerationService
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
