"""State contracts for the RAG graph."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

from typing_extensions import TypedDict

if TYPE_CHECKING:
    from cv_screener.rag.schema import (
        AnswerOutput,
        BriefAnswerOutput,
        FullCVOutput,
        PlannerOutput,
        ReviewOutput,
        RouteDecision,
        TargetedLookupOutput,
    )
    from cv_screener.retrieval.schema import RetrievedChunk
else:
    _rag_models = import_module("cv_screener.rag.schema")
    _retrieval_schema = import_module("cv_screener.retrieval.schema")
    AnswerOutput = _rag_models.AnswerOutput
    BriefAnswerOutput = _rag_models.BriefAnswerOutput
    FullCVOutput = _rag_models.FullCVOutput
    PlannerOutput = _rag_models.PlannerOutput
    ReviewOutput = _rag_models.ReviewOutput
    RouteDecision = _rag_models.RouteDecision
    TargetedLookupOutput = _rag_models.TargetedLookupOutput
    RetrievedChunk = _retrieval_schema.RetrievedChunk


class RAGState(TypedDict, total=False):
    """Shared state passed between RAG graph nodes."""

    user_query: str
    route: RouteDecision
    brief_answer: BriefAnswerOutput
    targeted_lookup: TargetedLookupOutput
    full_cv: FullCVOutput
    planner: PlannerOutput
    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]
    answer: AnswerOutput
    review: ReviewOutput
    final_text: str
    review_attempts: int
    nodes_executed: list[str]
