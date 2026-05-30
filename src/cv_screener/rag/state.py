"""State contracts for the RAG graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

if TYPE_CHECKING:
    from cv_screener.rag.models import (
        AnswerOutput,
        PlannerOutput,
        ReviewOutput,
        RouteDecision,
    )
    from cv_screener.retrieval.schema import RetrievedChunk


class RAGState(TypedDict, total=False):
    """Shared state passed between RAG graph nodes."""

    user_query: str
    route: RouteDecision
    planner: PlannerOutput
    retrieved_chunks: list[RetrievedChunk]
    reranked_chunks: list[RetrievedChunk]
    answer: AnswerOutput
    review: ReviewOutput
    final_text: str
    review_attempts: int
