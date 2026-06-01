"""Thin application service for running the RAG graph from the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from cv_screener.config import SQLiteSettings
from cv_screener.persistence import SQLiteLookupService
from cv_screener.rag.graph import CompiledRAGGraph, GraphDependencies, build_rag_graph
from cv_screener.rag.nodes import (
    LocalReranker,
    build_answer_model,
    build_brief_answer_model,
    build_planner_model,
    build_reviewer_model,
    build_router_model,
)
from cv_screener.rag.state import RAGState
from cv_screener.retrieval.hybrid import HybridRetriever

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from cv_screener.rag.state import RAGState


@dataclass(frozen=True)
class RAGQueryResult:
    """Minimal response contract for terminal query execution."""

    final_text: str
    state: RAGState


class RAGQueryService:
    """Run recruiter-style queries through the LangGraph runtime."""

    def __init__(
        self,
        *,
        graph: CompiledRAGGraph | None = None,
        dependencies: GraphDependencies | None = None,
        candidate_name_min_score: float = 0.72,
    ) -> None:
        """Bind a compiled graph and its concrete runtime dependencies."""
        self._graph = graph or build_rag_graph()
        self._dependencies = dependencies or GraphDependencies(
            router_model=build_router_model(),
            planner_model=build_planner_model(),
            brief_answer_model=build_brief_answer_model(),
            lookup_service=SQLiteLookupService(
                sqlite_path=Path(SQLiteSettings().sqlite_path),
                candidate_name_min_score=candidate_name_min_score,
            ),
            retriever=HybridRetriever(),
            reranker=LocalReranker(),
            answer_model=build_answer_model(),
            reviewer_model=build_reviewer_model(),
        )

    def run(self, query_text: str) -> RAGQueryResult:
        """Execute a single query and return the final text plus terminal state."""
        normalized_query = query_text.strip()
        if not normalized_query:
            msg = "query_text must be a non-empty string"
            raise ValueError(msg)

        state = cast(
            "RAGState",
            self._graph.invoke(
                {"user_query": normalized_query},
                context=self._dependencies,
            ),
        )

        final_text = state.get("final_text")
        if not isinstance(final_text, str) or not final_text.strip():
            msg = "RAG graph did not produce final_text"
            raise ValueError(msg)

        return RAGQueryResult(final_text=final_text, state=state)

    async def async_stream(
        self,
        query_text: str,
        *,
        conversation_context: str | None = None,
    ) -> AsyncIterator[tuple[str, dict[str, Any]]]:
        """Yield (node_name, state_update) as each graph node completes."""
        normalized_query = query_text.strip()
        if not normalized_query:
            msg = "query_text must be a non-empty string"
            raise ValueError(msg)

        input_state: RAGState = {"user_query": normalized_query}
        if conversation_context is not None and conversation_context.strip():
            input_state["conversation_context"] = conversation_context.strip()

        async for event in self._graph.astream(
            input_state,
            context=self._dependencies,
            stream_mode="updates",
        ):
            for node_name, state_update in event.items():
                yield node_name, state_update
