"""Generate and validate YAML CV content files."""

import re
from collections.abc import Callable
from enum import StrEnum
from pathlib import Path
from uuid import uuid4

from loguru import logger

from cv_screener.cv_generation.schema import CVProfile
from cv_screener.cv_generation.sources import (
    CVProfileSource,
    SeededCVProfileSource,
)
from cv_screener.cv_generation.yaml_io import load_cv_profile, write_cv_profile


class GenerationMode(StrEnum):
    """Supported CV generation modes."""

    SEEDED = "seeded"
    LLM = "llm"


class LogVerbosity(StrEnum):
    """Supported logging verbosity levels for generation commands."""

    TRACE = "trace"
    DEBUG = "debug"
    INFO = "info"
    SUCCESS = "success"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ProgressMode(StrEnum):
    """Supported progress display modes for generation commands."""

    AUTO = "auto"
    ON = "on"
    OFF = "off"


type ProgressCallback = Callable[[int, int], None]


def build_filename(profile: CVProfile) -> str:
    """Build a human-readable filename backed by the canonical candidate ID."""
    name_slug = slugify(profile.full_name)
    return f"{name_slug}-{profile.candidate_id}.yaml"


def slugify(value: str) -> str:
    """Convert a display string into a filesystem-friendly lowercase slug."""
    normalized = re.sub(r"[^a-z0-9]+", "-", value.lower())
    return normalized.strip("-")


class CVGenerationService:
    """Service for writing and validating CV content YAML files."""

    def __init__(
        self,
        output_dir: Path,
        *,
        profile_source: CVProfileSource | None = None,
    ) -> None:
        """Initialize the service with the target output directory."""
        self.output_dir = output_dir
        self.profile_source = profile_source or SeededCVProfileSource()

    def generate(
        self,
        count: int,
        progress_callback: ProgressCallback | None = None,
    ) -> list[Path]:
        """Generate a fixed number of validated CV content files."""
        logger.debug(
            "Generating CV content files",
            count=count,
            output_dir=str(self.output_dir),
            source_type=type(self.profile_source).__name__,
        )
        written_files: list[Path] = []
        for index in range(count):
            candidate_number = index + 1
            logger.debug(
                "Generating CV content draft",
                progress=f"{candidate_number}/{count}",
                candidate_number=candidate_number,
                total_candidates=count,
                source_type=type(self.profile_source).__name__,
            )
            draft_profile = self.profile_source.generate_draft(index=index)
            payload = {
                "candidate_id": str(uuid4()),
                **draft_profile.model_dump(mode="json"),
            }
            profile = CVProfile.model_validate(payload)
            output_path = self.output_dir / build_filename(profile)
            write_cv_profile(output_path, profile)
            written_files.append(output_path)
            if progress_callback is not None:
                progress_callback(candidate_number, count)
            logger.debug(
                "Generated and wrote CV content file",
                progress=f"{candidate_number}/{count}",
                candidate_number=candidate_number,
                total_candidates=count,
                candidate_id=str(profile.candidate_id),
                output_path=str(output_path),
            )
        logger.debug(
            "Generated CV content files",
            count=len(written_files),
            output_dir=str(self.output_dir),
        )
        return written_files

    def validate_directory(self, directory: Path) -> list[Path]:
        """Validate every YAML CV content file in a directory."""
        logger.info("Validating CV content files", input_dir=str(directory))
        validated_files: list[Path] = []
        for path in sorted(directory.glob("*.yaml")):
            logger.debug("Validating CV content file", path=str(path))
            load_cv_profile(path)
            validated_files.append(path)
        logger.info(
            "Validated CV content files",
            count=len(validated_files),
            input_dir=str(directory),
        )
        return validated_files
