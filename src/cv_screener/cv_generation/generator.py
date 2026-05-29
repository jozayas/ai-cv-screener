"""Generate and validate YAML CV content files."""

import re
from pathlib import Path
from uuid import uuid4

from loguru import logger

from cv_screener.cv_generation.schema import CVProfile
from cv_screener.cv_generation.yaml_io import load_cv_profile, write_cv_profile

# This is intentionally deterministic seed data for the first slice.
# LLM-backed generation will replace or extend this in a later increment.
SEED_PROFILES: list[dict] = [
    {
        "full_name": "Marta Alvarez",
        "email": "marta.alvarez@example.com",
        "phone": "+34 600 111 111",
        "location": "Barcelona, Spain",
        "professional_summary": "Backend engineer with experience building APIs, data pipelines, and internal tooling.",
        "skills": ["Python", "FastAPI", "PostgreSQL", "Docker", "AWS"],
        "experience": [
            {
                "company": "NovaStack",
                "role": "Backend Engineer",
                "start_date": "2021-03-01",
                "end_date": None,
                "summary": "Built backend services for workflow automation.",
                "highlights": [
                    "Implemented REST APIs for internal products.",
                    "Reduced batch processing time by optimizing SQL queries.",
                ],
                "technologies": ["Python", "FastAPI", "PostgreSQL"],
            }
        ],
        "education": [
            {
                "institution": "Universitat Politecnica de Catalunya",
                "degree": "BSc",
                "field_of_study": "Computer Engineering",
                "graduation_year": 2020,
            }
        ],
    },
    {
        "full_name": "Daniel Romero",
        "email": "daniel.romero@example.com",
        "phone": "+34 600 222 222",
        "location": "Madrid, Spain",
        "professional_summary": "Full-stack developer focused on product delivery, frontend quality, and developer experience.",
        "skills": ["TypeScript", "React", "Node.js", "GraphQL", "Playwright"],
        "experience": [
            {
                "company": "PixelForge",
                "role": "Full-stack Developer",
                "start_date": "2020-06-01",
                "end_date": None,
                "summary": "Delivered customer-facing features across web applications.",
                "highlights": [
                    "Shipped React interfaces for analytics workflows.",
                    "Added end-to-end test coverage for release-critical flows.",
                ],
                "technologies": ["TypeScript", "React", "Node.js"],
            }
        ],
        "education": [
            {
                "institution": "Universidad Carlos III de Madrid",
                "degree": "BEng",
                "field_of_study": "Software Engineering",
                "graduation_year": 2019,
            }
        ],
    },
    {
        "full_name": "Lucia Moreno",
        "email": "lucia.moreno@example.com",
        "phone": "+34 600 333 333",
        "location": "Valencia, Spain",
        "professional_summary": "Data engineer with experience in ETL pipelines, analytics modeling, and platform reliability.",
        "skills": ["Python", "dbt", "Airflow", "BigQuery", "Terraform"],
        "experience": [
            {
                "company": "SignalRiver",
                "role": "Data Engineer",
                "start_date": "2022-01-10",
                "end_date": None,
                "summary": "Maintained data ingestion and transformation pipelines for analytics teams.",
                "highlights": [
                    "Modeled warehouse tables for self-serve reporting.",
                    "Improved observability for scheduled data jobs.",
                ],
                "technologies": ["Python", "Airflow", "BigQuery"],
            }
        ],
        "education": [
            {
                "institution": "Universitat de Valencia",
                "degree": "MSc",
                "field_of_study": "Data Science",
                "graduation_year": 2021,
            }
        ],
    },
]


class CVGenerationService:
    """Service for writing and validating CV content YAML files."""

    def __init__(self, output_dir: Path) -> None:
        """Initialize the service with the target output directory."""
        self.output_dir = output_dir

    def generate(self, count: int) -> list[Path]:
        """Generate a fixed number of validated seeded CV content files."""
        logger.bind(
            count=count,
            output_dir=str(self.output_dir),
            mode="seeded-placeholder",
        ).info("Generating seeded CV content files")
        written_files: list[Path] = []
        for seed_profile in self._iter_seed_profiles(count):
            payload = {
                "candidate_id": str(uuid4()),
                **seed_profile,
            }
            profile = CVProfile.model_validate(payload)
            output_path = self.output_dir / self._build_filename(profile)
            write_cv_profile(output_path, profile)
            written_files.append(output_path)
        logger.bind(count=len(written_files), output_dir=str(self.output_dir)).info(
            "Generated CV content files"
        )
        return written_files

    def _iter_seed_profiles(self, count: int) -> list[dict]:
        """Return the seeded profiles needed to satisfy a generation request."""
        return [SEED_PROFILES[index % len(SEED_PROFILES)] for index in range(count)]

    def _build_filename(self, profile: CVProfile) -> str:
        """Build a human-readable filename backed by the canonical candidate ID."""
        name_slug = self._slugify(profile.full_name)
        return f"{name_slug}-{profile.candidate_id}.yaml"

    def _slugify(self, value: str) -> str:
        """Convert a display string into a filesystem-friendly lowercase slug."""
        normalized = re.sub(r"[^a-z0-9]+", "-", value.lower())
        return normalized.strip("-")

    def validate_directory(self, directory: Path) -> list[Path]:
        """Validate every YAML CV content file in a directory."""
        logger.bind(input_dir=str(directory)).info("Validating CV content files")
        validated_files: list[Path] = []
        for path in sorted(directory.glob("*.yaml")):
            load_cv_profile(path)
            validated_files.append(path)
        logger.bind(count=len(validated_files), input_dir=str(directory)).info(
            "Validated CV content files"
        )
        return validated_files
