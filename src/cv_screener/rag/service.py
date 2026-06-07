"""Thin application service for running the RAG graph from the CLI."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

from cv_screener.config import AppSettings
from cv_screener.persistence.lookup import SQLiteLookupService
from cv_screener.rag.graph import CompiledRAGGraph, GraphDependencies, build_rag_graph
from cv_screener.rag.nodes.answer import build_answer_model
from cv_screener.rag.nodes.brief_answer import build_brief_answer_model
from cv_screener.rag.nodes.planner import build_planner_model
from cv_screener.rag.nodes.rerank import LocalReranker, ScoreReranker
from cv_screener.rag.nodes.review import build_reviewer_model
from cv_screener.rag.nodes.route import build_router_model
from cv_screener.rag.state import RAGState

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from cv_screener.rag.nodes.retrieve import RetrieverProtocol
    from cv_screener.rag.state import RAGState
    from cv_screener.retrieval.schema import RetrievedChunk


@dataclass(frozen=True)
class RAGQueryResult:
    """Minimal response contract for terminal query execution."""

    final_text: str
    state: RAGState


class _LazyHybridRetriever:
    """Load the embedding-backed retriever only when retrieval is needed."""

    def __init__(self) -> None:
        self._retriever: RetrieverProtocol | None = None

    def retrieve(self, query_text: str) -> list[RetrievedChunk]:
        if self._retriever is None:
            module = import_module("cv_screener.retrieval.hybrid")
            self._retriever = module.HybridRetriever()
        return self._retriever.retrieve(query_text)


class RAGQueryService:
    """Run recruiter-style queries through the LangGraph runtime."""

    def __init__(
        self,
        *,
        graph: CompiledRAGGraph | None = None,
        dependencies: GraphDependencies | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        """Bind a compiled graph and its concrete runtime dependencies."""
        self._graph = graph or build_rag_graph()
        if dependencies is not None:
            self._dependencies = dependencies
            return

        resolved_settings = settings or AppSettings()
        rag_settings = resolved_settings.rag
        self._dependencies = GraphDependencies(
            router_model=build_router_model(rag_settings),
            planner_model=build_planner_model(rag_settings),
            brief_answer_model=build_brief_answer_model(rag_settings),
            lookup_service=SQLiteLookupService(
                sqlite_path=Path(resolved_settings.sqlite.sqlite_path),
                candidate_name_min_score=resolved_settings.lookup.candidate_name_min_score,
            ),
            retriever=_LazyHybridRetriever(),
            reranker=(
                LocalReranker()
                if rag_settings.rag_enable_cross_encoder_rerank
                else ScoreReranker()
            ),
            answer_model=build_answer_model(rag_settings),
            reviewer_model=build_reviewer_model(rag_settings),
            enable_llm_review=rag_settings.rag_enable_llm_review,
            max_retrieval_queries=rag_settings.rag_max_retrieval_queries,
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
