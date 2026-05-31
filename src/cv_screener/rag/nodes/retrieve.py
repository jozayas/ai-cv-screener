"""Deterministic retrieval node over the hybrid retriever."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

from cv_screener.rag.schema import PlannerOutput

if TYPE_CHECKING:
    from cv_screener.rag.state import RAGState
    from cv_screener.retrieval.schema import RetrievedChunk


class RetrieverProtocol(Protocol):
    """Behavior required from the deterministic retrieval node."""

    def retrieve(self, query_text: str) -> list[RetrievedChunk]:
        """Return ranked chunks for a retrieval-oriented query."""
        ...


def retrieve_node(
    state: RAGState,
    *,
    retriever: RetrieverProtocol,
) -> dict[str, list[RetrievedChunk]]:
    """Fetch retrieved chunks for the planner's primary query."""
    planner = state.get("planner")
    if not isinstance(planner, PlannerOutput):
        msg = "retrieval state must include planner output"
        raise TypeError(msg)
    return {"retrieved_chunks": retriever.retrieve(planner.primary_query)}
