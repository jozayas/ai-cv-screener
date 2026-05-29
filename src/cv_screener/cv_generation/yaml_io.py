"""YAML serialization helpers for CV content files."""

from pathlib import Path

import yaml

from cv_screener.cv_generation.schema import CVProfile


def load_cv_profile(path: Path) -> CVProfile:
    """Load and validate a CV profile from YAML."""
    with path.open("r", encoding="utf-8") as file:
        raw_data = yaml.safe_load(file) or {}
    return CVProfile.model_validate(raw_data)


def write_cv_profile(path: Path, profile: CVProfile) -> None:
    """Write a validated CV profile to YAML."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(
            profile.model_dump(mode="json"),
            file,
            sort_keys=False,
            allow_unicode=False,
        )
