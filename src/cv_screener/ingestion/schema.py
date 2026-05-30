"""Pydantic models for parsed PDF content."""

from pathlib import Path

from pydantic import BaseModel, Field


class ParsedPage(BaseModel):
    """A single page extracted from a PDF document."""

    page_number: int = Field(ge=1, description="1-indexed page number.")
    markdown: str = Field(
        min_length=1, description="Markdown text extracted from the page."
    )


class ParsedCV(BaseModel):
    """A fully parsed CV document with source metadata."""

    source_path: Path = Field(description="Original PDF file path.")
    filename: str = Field(min_length=1, description="Stem filename for citation.")
    title: str = Field(default="", description="PDF document title if available.")
    pages: list[ParsedPage] = Field(
        min_length=1, description="Ordered pages of the CV."
    )

    @property
    def full_text(self) -> str:
        """Concatenate all page markdown texts."""
        return "\n\n".join(page.markdown for page in self.pages)


class Chunk(BaseModel):
    """A section-aware chunk of a CV document for embedding and retrieval."""

    source_filename: str = Field(
        min_length=1, description="Stem filename of source CV."
    )
    page_number: int = Field(
        ge=1, description="1-indexed page number where chunk starts."
    )
    section: str = Field(
        min_length=1, description="Section heading this chunk belongs to."
    )
    text: str = Field(min_length=1, description="Chunk text content.")


class ChunkConfig(BaseModel):
    """Configuration for section-aware CV chunking."""

    similarity_threshold: float = Field(
        default=0.6,
        ge=0.0,
        le=1.0,
        description="Minimum similarity for mapping a heading to a canonical section.",
    )
    model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        min_length=1,
        description="Embedding model used to classify section headings.",
    )
    canonical_sections: list[str] = Field(
        default_factory=lambda: [
            "SUMMARY",
            "EXPERIENCE",
            "SKILLS",
            "EDUCATION",
            "PROJECTS",
            "CERTIFICATIONS",
            "LANGUAGES",
            "INTERESTS",
            "PUBLICATIONS",
            "AWARDS",
            "VOLUNTEER",
            "OBJECTIVE",
            "PROFILE",
            "REFERENCES",
            "ACHIEVEMENTS",
            "RESEARCH",
            "LEADERSHIP",
        ],
        min_length=1,
        description="Canonical section labels used for semantic heading classification.",
    )
