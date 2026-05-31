"""LangGraph wiring for the RAG runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph

from cv_screener.rag.answerer import answerer_node
from cv_screener.rag.brief_answer import brief_answer_node
from cv_screener.rag.models import (
    AnswerOutput,
    BriefAnswerOutput,
    PlannerOutput,
    RouteDecision,
    RouteTarget,
)
from cv_screener.rag.planner import planner_node
from cv_screener.rag.reranker import RerankerProtocol, reranker_node
from cv_screener.rag.reviewer import reviewer_node
from cv_screener.rag.router import next_node_for_route, router_node
from cv_screener.rag.state import RAGState

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable
    from langgraph.runtime import Runtime

    from cv_screener.rag.models import ReviewOutput
    from cv_screener.retrieval.schema import RetrievedChunk


class CompiledRAGGraph(Protocol):
    """Minimal compiled graph protocol used by the current slice."""

    def invoke(self, state: RAGState, *, context: GraphDependencies) -> RAGState:
        """Run the graph to completion for a single state input."""
        ...


class RetrieverProtocol(Protocol):
    """Behavior required from the deterministic retrieval node."""

    def retrieve(self, query_text: str) -> list[RetrievedChunk]:
        """Return ranked chunks for a retrieval-oriented query."""
        ...


@dataclass(frozen=True)
class GraphDependencies:
    """Immutable runtime dependencies for graph nodes."""

    router_model: Runnable[LanguageModelInput, RouteDecision]
    planner_model: Runnable[LanguageModelInput, PlannerOutput]
    brief_answer_model: Runnable[LanguageModelInput, BriefAnswerOutput]
    retriever: RetrieverProtocol
    reranker: RerankerProtocol
    answer_model: Runnable[LanguageModelInput, AnswerOutput]
    reviewer_model: Runnable[LanguageModelInput, ReviewOutput]


def build_rag_graph(
) -> CompiledRAGGraph:
    """Build the RAG runtime graph."""
    graph = StateGraph(cast("Any", RAGState), context_schema=GraphDependencies)
    _ = graph.add_node("router", router_graph_node)
    _ = graph.add_node("brief_answer", brief_answer_graph_node)
    _ = graph.add_node("planner", planner_graph_node)
    _ = graph.add_node("retrieve", retrieve_graph_node)
    _ = graph.add_node("rerank", rerank_graph_node)
    _ = graph.add_node("answer", answer_graph_node)
    _ = graph.add_node("review", review_graph_node)
    _ = graph.add_node("finalize", finalize_node)

    _ = graph.add_edge(START, "router")
    _ = graph.add_conditional_edges("router", _next_node_from_state)
    _ = graph.add_edge("brief_answer", "finalize")
    _ = graph.add_edge("planner", "retrieve")
    _ = graph.add_edge("retrieve", "rerank")
    _ = graph.add_edge("rerank", "answer")
    _ = graph.add_edge("answer", "review")
    _ = graph.add_edge("review", "finalize")
    _ = graph.add_edge("finalize", END)

    return cast("CompiledRAGGraph", graph.compile())


def router_graph_node(
    state: RAGState,
    runtime: Runtime[GraphDependencies],
) -> dict[str, RouteDecision]:
    """Graph adapter for the router node."""
    return router_node(
        state,
        config=get_config(),
        model=runtime.context.router_model,
    )


def brief_answer_graph_node(
    state: RAGState,
    runtime: Runtime[GraphDependencies],
) -> dict[str, BriefAnswerOutput]:
    """Graph adapter for the brief-answer node."""
    return brief_answer_node(
        state,
        config=get_config(),
        model=runtime.context.brief_answer_model,
    )


def planner_graph_node(
    state: RAGState,
    runtime: Runtime[GraphDependencies],
) -> dict[str, PlannerOutput]:
    """Graph adapter for the planner node."""
    return planner_node(
        state,
        config=get_config(),
        model=runtime.context.planner_model,
    )


def retrieve_graph_node(
    state: RAGState,
    runtime: Runtime[GraphDependencies],
) -> dict[str, list[RetrievedChunk]]:
    """Graph adapter for the retrieval node."""
    return retrieve_node(state, retriever=runtime.context.retriever)


def rerank_graph_node(
    state: RAGState,
    runtime: Runtime[GraphDependencies],
) -> dict[str, list[RetrievedChunk]]:
    """Graph adapter for the rerank node."""
    return reranker_node(state, reranker=runtime.context.reranker)


def answer_graph_node(
    state: RAGState,
    runtime: Runtime[GraphDependencies],
) -> dict[str, AnswerOutput]:
    """Graph adapter for the grounded answer node."""
    return answerer_node(
        state,
        config=get_config(),
        model=runtime.context.answer_model,
    )


def review_graph_node(
    state: RAGState,
    runtime: Runtime[GraphDependencies],
) -> dict[str, object]:
    """Graph adapter for the groundedness review node."""
    return reviewer_node(
        state,
        config=get_config(),
        model=runtime.context.reviewer_model,
    )


def retrieve_node(
    state: RAGState,
    *,
    retriever: RetrieverProtocol,
) -> dict[str, list[RetrievedChunk]]:
    """Deterministic retrieval node over the existing hybrid retriever."""
    planner = state.get("planner")
    if not isinstance(planner, PlannerOutput):
        msg = "retrieval state must include planner output"
        raise TypeError(msg)
    return {"retrieved_chunks": retriever.retrieve(planner.primary_query)}


def finalize_node(state: RAGState) -> dict[str, str]:
    """Produce the user-visible response for terminal graph routes."""
    route = state.get("route")
    if not isinstance(route, RouteDecision):
        msg = "finalize state must include a route decision"
        raise TypeError(msg)

    if route.route is not RouteTarget.CV_QUERY:
        brief_answer = state.get("brief_answer")
        if not isinstance(brief_answer, BriefAnswerOutput):
            return {}
        return {"final_text": brief_answer.text}

    answer = state.get("answer")
    if not isinstance(answer, AnswerOutput):
        return {}
    return {"final_text": _format_answer(answer)}


def _next_node_from_state(state: RAGState) -> Literal["brief_answer", "planner"]:
    route = state.get("route")
    if not isinstance(route, RouteDecision):
        msg = "router state must include a route decision"
        raise TypeError(msg)
    return next_node_for_route(route)


def _format_answer(answer: AnswerOutput) -> str:
    if answer.abstained:
        return answer.answer

    seen: set[tuple[str, int, str]] = set()
    citation_lines: list[str] = []
    for citation in answer.citations:
        citation_key = (citation.source_file, citation.page, citation.section)
        if citation_key in seen:
            continue
        seen.add(citation_key)
        citation_lines.append(
            f"- {citation.source_file} (page {citation.page}, {citation.section})"
        )

    return "\n\n".join([answer.answer, "Sources:\n" + "\n".join(citation_lines)])
