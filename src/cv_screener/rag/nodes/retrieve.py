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
    """Fetch retrieved chunks for the planner's primary query and alternate queries."""
    planner = state.get("planner")
    if not isinstance(planner, PlannerOutput):
        msg = "retrieval state must include planner output"
        raise TypeError(msg)

    queries = [planner.primary_query, *planner.alternate_queries]
    all_chunks: list[RetrievedChunk] = []
    for query in queries:
        all_chunks.extend(retriever.retrieve(query))

    seen: set[tuple[str, int, str]] = set()
    merged: list[RetrievedChunk] = []
    for chunk in sorted(all_chunks, key=lambda c: c.score, reverse=True):
        key = (chunk.source_file, chunk.page, chunk.section)
        if key not in seen:
            seen.add(key)
            merged.append(chunk)

    for rank, chunk in enumerate(merged, start=1):
        chunk.rank = rank

    return {"retrieved_chunks": merged}
