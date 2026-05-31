"""Generate and validate YAML CV content files."""

import re
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from enum import StrEnum
from pathlib import Path
from typing import Protocol
from uuid import uuid4

from loguru import logger

from cv_screener.cv_generation.content.schema import CVProfile, CVProfileDraft
from cv_screener.cv_generation.content.yaml_io import (
    load_cv_profile,
    write_cv_profile,
)


class GenerationMode(StrEnum):
    """Supported CV generation modes."""

    SEEDED = "seeded"
    LLM = "llm"


type ProgressCallback = Callable[[int, int], None]


class CVProfileSource(Protocol):
    """Profile source that can generate draft CV payloads."""

    @property
    def max_concurrency(self) -> int:
        """Maximum safe parallelism for this source."""
        ...

    def generate_draft(self, *, index: int) -> CVProfileDraft:
        """Generate a single draft CV payload."""
        ...


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
        if profile_source is None:
            from cv_screener.cv_generation.content.sources import (  # noqa: PLC0415
                SeededCVProfileSource,
            )

            profile_source = SeededCVProfileSource()
        self.profile_source = profile_source

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
        worker_count = self._resolve_worker_count(count)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            written_files = list(executor.map(self._generate_one, range(count)))
        if progress_callback is not None:
            for candidate_number in range(1, count + 1):
                progress_callback(candidate_number, count)
        logger.debug(
            "Generated CV content files",
            count=len(written_files),
            output_dir=str(self.output_dir),
        )
        return written_files

    def _resolve_worker_count(self, count: int) -> int:
        source_cap = getattr(self.profile_source, "max_concurrency", 1)
        return max(1, min(count, int(source_cap)))

    def _generate_one(self, index: int) -> Path:
        candidate_number = index + 1
        logger.debug(
            "Generating CV content draft",
            candidate_number=candidate_number,
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
        logger.debug(
            "Generated and wrote CV content file",
            candidate_number=candidate_number,
            candidate_id=str(profile.candidate_id),
            output_path=str(output_path),
        )
        return output_path

    def validate_directory(self, directory: Path) -> list[Path]:
        """Validate every YAML CV content file in a directory."""
        logger.info("Validating CV content files", input_dir=str(directory))
        validated_files = self.validate_files(sorted(directory.glob("*.yaml")))
        logger.info(
            "Validated CV content files",
            count=len(validated_files),
            input_dir=str(directory),
        )
        return validated_files

    def validate_files(self, paths: list[Path]) -> list[Path]:
        """Validate specific YAML CV content files."""
        validated_files: list[Path] = []
        for path in paths:
            logger.debug("Validating CV content file", path=str(path))
            load_cv_profile(path)
            validated_files.append(path)
        return validated_files
