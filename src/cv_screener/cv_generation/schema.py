"""Pydantic schema for generated CV content."""

from datetime import date
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator


class ExperienceEntry(BaseModel):
    """Employment history entry for a candidate."""

    company: str = Field(min_length=1)
    role: str = Field(min_length=1)
    start_date: date
    end_date: date | None = None
    summary: str = Field(min_length=1)
    highlights: list[str] = Field(default_factory=list)
    technologies: list[str] = Field(default_factory=list)

    @field_validator("end_date")
    @classmethod
    def validate_dates(cls, end_date: date | None, info: ValidationInfo) -> date | None:
        """Ensure experience end dates do not precede start dates."""
        start_date = info.data.get("start_date")
        if start_date and end_date and end_date < start_date:
            message = "end_date must be on or after start_date"
            raise ValueError(message)
        return end_date


class EducationEntry(BaseModel):
    """Education entry for a candidate."""

    institution: str = Field(min_length=1)
    degree: str = Field(min_length=1)
    field_of_study: str = Field(min_length=1)
    graduation_year: int = Field(ge=1900, le=2100)


class CVProfile(BaseModel):
    """Full validated YAML payload for a generated CV."""

    model_config = ConfigDict(extra="forbid")

    candidate_id: UUID
    full_name: str = Field(min_length=1)
    email: str = Field(min_length=1)
    phone: str = Field(min_length=1)
    location: str = Field(min_length=1)
    professional_summary: str = Field(min_length=1)
    skills: list[str] = Field(min_length=1)
    experience: list[ExperienceEntry] = Field(min_length=1)
    education: list[EducationEntry] = Field(min_length=1)
