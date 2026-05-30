"""Structured models for the RAG graph."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field, model_validator


class RouteTarget(StrEnum):
    """Supported high-level routes from the router node."""

    SMALL_TALK = "small_talk"
    CV_QUERY = "cv_query"
    NEEDS_CLARIFICATION = "needs_clarification"


class RouteDecision(BaseModel):
    """Structured router output used to branch the RAG graph."""

    route: RouteTarget
    reasoning: str = Field(min_length=1)
    small_talk_response: str | None = Field(default=None, min_length=1)
    clarification_question: str | None = Field(default=None, min_length=1)

    @model_validator(mode="after")
    def validate_route_payload(self) -> RouteDecision:
        """Require the route-specific text fields needed by downstream nodes."""
        if (
            self.route is RouteTarget.SMALL_TALK
            and self.small_talk_response is None
        ):
            msg = "small_talk_response is required when route=small_talk"
            raise ValueError(msg)
        if (
            self.route is RouteTarget.NEEDS_CLARIFICATION
            and self.clarification_question is None
        ):
            msg = (
                "clarification_question is required when "
                "route=needs_clarification"
            )
            raise ValueError(msg)
        return self


class SearchFacets(BaseModel):
    """Lightweight structured filters extracted from a recruiter-style query."""

    skills: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)
    seniority: str | None = Field(default=None, min_length=1)
    education: str | None = Field(default=None, min_length=1)


class PlannerOutput(BaseModel):
    """Structured query rewrite result for retrieval."""

    primary_query: str = Field(min_length=1)
    alternate_queries: list[str] = Field(default_factory=list, max_length=2)
    facets: SearchFacets = Field(default_factory=SearchFacets)


class AnswerCitation(BaseModel):
    """Citation metadata carried into final answers."""

    source_file: str = Field(min_length=1)
    page: int = Field(ge=1)
    section: str = Field(min_length=1)


class AnswerOutput(BaseModel):
    """Structured answer draft produced before final formatting."""

    answer: str = Field(min_length=1)
    citations: list[AnswerCitation] = Field(default_factory=list)
    abstained: bool = Field(default=False)


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
