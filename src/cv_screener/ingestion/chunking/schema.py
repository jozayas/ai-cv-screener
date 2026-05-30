"""Pydantic models for retrieval chunking."""

from pydantic import BaseModel, Field, model_validator


class Chunk(BaseModel):
    """A retrieval-ready chunk of a CV document."""

    candidate_name: str | None = Field(
        default=None,
        description="Candidate display name extracted from the PDF source when available.",
    )
    source_file: str = Field(
        min_length=1,
        description="Source PDF filename for citations and UI display.",
    )
    document_title: str = Field(
        default="",
        description="PDF document title if available for citation display.",
    )
    page: int = Field(
        ge=1, description="1-indexed page number where chunk starts."
    )
    chunk_index: int = Field(
        ge=0,
        description="0-indexed chunk position within the parsed CV.",
    )
    section: str = Field(
        min_length=1,
        description="Section heading this chunk belongs to after section classification.",
    )
    detected_skills: list[str] = Field(
        default_factory=list,
        description="Skills detected from the chunk text.",
    )
    detected_companies: list[str] = Field(
        default_factory=list,
        description="Company names detected from the chunk text.",
    )
    detected_universities: list[str] = Field(
        default_factory=list,
        description="Universities or educational institutions detected from the chunk text.",
    )
    email_addresses: list[str] = Field(
        default_factory=list,
        description="Email addresses detected from the chunk text.",
    )
    phone_numbers: list[str] = Field(
        default_factory=list,
        description="Phone numbers detected from the chunk text.",
    )
    linkedin_urls: list[str] = Field(
        default_factory=list,
        description="LinkedIn URLs detected from the chunk text.",
    )
    github_urls: list[str] = Field(
        default_factory=list,
        description="GitHub URLs detected from the chunk text.",
    )
    text: str = Field(min_length=1, description="Chunk text content.")


class ChunkConfig(BaseModel):
    """Configuration for section-aware semantic CV chunking."""

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
    chunk_size: int = Field(
        default=600,
        ge=100,
        description="Maximum character length for semantic sub-chunks.",
    )
    chunk_overlap: int = Field(
        default=80,
        ge=0,
        description="Character overlap between semantic sub-chunks.",
    )
    semantic_similarity_threshold: float = Field(
        default=0.55,
        ge=0.0,
        le=1.0,
        description="Minimum adjacent similarity to keep semantic units in the same chunk.",
    )
    gliner_model_name: str = Field(
        default="urchade/gliner_small-v2.1",
        min_length=1,
        description="GLiNER model used for chunk metadata extraction.",
    )
    gliner_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence threshold used for GLiNER entity extraction.",
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

    @model_validator(mode="after")
    def validate_chunk_window(self) -> "ChunkConfig":
        """Ensure semantic chunk overlap is strictly smaller than chunk size."""
        if self.chunk_overlap >= self.chunk_size:
            message = "chunk_overlap must be smaller than chunk_size"
            raise ValueError(message)
        return self
