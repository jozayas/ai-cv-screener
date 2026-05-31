"""Thin application service for running the RAG graph from the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from cv_screener.rag.graph import CompiledRAGGraph, GraphDependencies, build_rag_graph
from cv_screener.rag.nodes import (
    LocalReranker,
    build_answer_model,
    build_brief_answer_model,
    build_planner_model,
    build_reviewer_model,
    build_router_model,
)
from cv_screener.retrieval.hybrid import HybridRetriever

if TYPE_CHECKING:
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
    ) -> None:
        """Bind a compiled graph and its concrete runtime dependencies."""
        self._graph = graph or build_rag_graph()
        self._dependencies = dependencies or GraphDependencies(
            router_model=build_router_model(),
            planner_model=build_planner_model(),
            brief_answer_model=build_brief_answer_model(),
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
