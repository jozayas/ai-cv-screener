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
