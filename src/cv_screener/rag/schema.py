"""Structured models for the RAG graph."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class RouteTarget(StrEnum):
    """Supported high-level routes from the router node."""

    SMALL_TALK = "small_talk"
    CV_QUERY = "cv_query"
    TARGETED_LOOKUP = "targeted_lookup"
    FULL_CV = "full_cv"
    NEEDS_CLARIFICATION = "needs_clarification"


class RouteDecision(BaseModel):
    """Structured router output used to branch the RAG graph."""

    route: RouteTarget
    reasoning: str = Field(min_length=1)


class SearchFacets(BaseModel):
    """Lightweight structured filters extracted from a recruiter-style query."""

    skills: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    seniority: str | None = Field(default=None, min_length=1)
    education: str | None = Field(default=None, min_length=1)

    @model_validator(mode="before")
    @classmethod
    def coerce_null_lists(cls, data: dict) -> dict:
        """Coerce None list fields to empty lists before validation."""
        if not isinstance(data, dict):
            return data
        for field in ("skills", "languages", "roles"):
            if data.get(field) is None:
                data[field] = []
        return data


class PlannerOutput(BaseModel):
    """Structured query rewrite result for retrieval."""

    primary_query: str = Field(min_length=1)
    alternate_queries: list[str] = Field(default_factory=list, max_length=2)
    facets: SearchFacets = Field(default_factory=SearchFacets)

    @model_validator(mode="before")
    @classmethod
    def coerce_null_alternate_queries(cls, data: dict) -> dict:
        """Coerce None alternate_queries to empty list before validation."""
        if not isinstance(data, dict):
            return data
        if data.get("alternate_queries") is None:
            data["alternate_queries"] = []
        return data


class AnswerCitation(BaseModel):
    """Citation metadata carried into final answers."""

    rank: int = Field(ge=1, description="Reranker rank used as the citation number.")
    candidate_name: str | None = Field(default=None, min_length=1)
    source_file: str = Field(min_length=1)
    page: int = Field(ge=1)
    section: str = Field(min_length=1)


class AnswerOutput(BaseModel):
    """Structured answer draft produced before final formatting."""

    answer: str = Field(min_length=1)
    citations: list[AnswerCitation] = Field(default_factory=list)
    abstained: bool = Field(default=False)

    @model_validator(mode="before")
    @classmethod
    def strip_citations_when_abstained(cls, data: dict) -> dict:
        """Strip citations when abstained is explicitly true."""
        if not isinstance(data, dict):
            return data
        if data.get("abstained") is True:
            data["citations"] = []
        return data

    @model_validator(mode="after")
    def validate_answer_payload(self) -> AnswerOutput:
        """Require citations for substantive answers and none for abstentions."""
        if self.abstained:
            if self.citations:
                msg = "abstained answers must not include citations"
                raise ValueError(msg)
            return self
        if not self.citations:
            msg = "citations are required when abstained=False"
            raise ValueError(msg)
        return self


class BriefAnswerOutput(BaseModel):
    """Structured direct reply for non-retrieval turns."""

    text: str = Field(min_length=1)


class TargetedLookupOutput(BaseModel):
    """Deterministic SQLite lookup result used before hydration."""

    candidate_ids: list[str] = Field(default_factory=list)
    candidate_names: list[str] = Field(default_factory=list)
    sections: list[str] = Field(default_factory=list)
    evidence_term: str | None = Field(default=None, min_length=1)
    response_mode: str = Field(default="profile")
    clarification_message: str | None = Field(default=None, min_length=1)
    fallback_to_semantic: bool = Field(default=False)


class FullCVOutput(BaseModel):
    """Resolved CV document metadata for direct-return requests."""

    candidate_name: str = Field(min_length=1)
    source_file: str = Field(min_length=1)
    document_title: str = Field(min_length=1)
    pdf_path: str = Field(min_length=1)
    parsed_markdown: str = Field(min_length=1)


class ReviewVerdict(StrEnum):
    """Supported reviewer outcomes."""

    APPROVE = "approve"
    REVISE = "revise"
    ABSTAIN = "abstain"


class ReviewOutput(BaseModel):
    """Structured groundedness review output."""

    verdict: ReviewVerdict
    reasoning: str = Field(min_length=1)
    revised_answer: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_review_payload(self) -> ReviewOutput:
        """Require a revised answer only for revise verdicts."""
        if self.verdict is ReviewVerdict.REVISE and self.revised_answer is None:
            msg = "revised_answer is required when verdict=revise"
            raise ValueError(msg)
        if self.verdict is not ReviewVerdict.REVISE and self.revised_answer is not None:
            msg = "revised_answer is only allowed when verdict=revise"
            raise ValueError(msg)
        return self
