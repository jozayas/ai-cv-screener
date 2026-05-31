"""LangGraph wiring for the RAG runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, cast

from langgraph.config import get_config
from langgraph.graph import END, START, StateGraph

from cv_screener.rag.nodes.answer import answerer_node
from cv_screener.rag.nodes.brief_answer import brief_answer_node
from cv_screener.rag.nodes.finalize import finalize_node
from cv_screener.rag.nodes.planner import planner_node
from cv_screener.rag.nodes.rerank import RerankerProtocol, reranker_node
from cv_screener.rag.nodes.retrieve import retrieve_node as retrieve_state_node
from cv_screener.rag.nodes.review import reviewer_node
from cv_screener.rag.nodes.route import next_node_for_route, router_node
from cv_screener.rag.schema import (
    AnswerOutput,
    BriefAnswerOutput,
    PlannerOutput,
    ReviewOutput,
    ReviewVerdict,
    RouteDecision,
)
from cv_screener.rag.state import RAGState

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable
    from langgraph.graph.state import CompiledStateGraph
    from langgraph.runtime import Runtime

    from cv_screener.rag.nodes.retrieve import RetrieverProtocol


class CompiledRAGGraph(Protocol):
    """Minimal compiled graph protocol used by the current slice."""

    def invoke(self, state: RAGState, *, context: GraphDependencies) -> RAGState:
        """Run the graph to completion for a single state input."""
        ...

    def astream(
        self,
        state: RAGState,
        *,
        context: GraphDependencies | None = None,
        stream_mode: str | None = "updates",
    ) -> AsyncIterator[dict[str, Any]]:
        """Stream per-node updates from the graph."""
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


def build_rag_graph() -> CompiledStateGraph[Any, GraphDependencies, Any, Any]:
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
    _ = graph.add_conditional_edges("review", _next_node_after_review)
    _ = graph.add_edge("finalize", END)

    return graph.compile()


def _executed(state: RAGState, name: str) -> list[str]:
    """Append a node name to the execution trace."""
    return [*state.get("nodes_executed", []), name]


def router_graph_node(
    state: RAGState,
    runtime: Runtime[GraphDependencies],
) -> dict[str, object]:
    """Graph adapter for the router node."""
    return cast(
        "dict[str, object]",
        {
            **router_node(state, config=get_config(), model=runtime.context.router_model),
            "nodes_executed": _executed(state, "router"),
        },
    )


def brief_answer_graph_node(
    state: RAGState,
    runtime: Runtime[GraphDependencies],
) -> dict[str, object]:
    """Graph adapter for the brief-answer node."""
    return cast(
        "dict[str, object]",
        {
            **brief_answer_node(
                state, config=get_config(), model=runtime.context.brief_answer_model
            ),
            "nodes_executed": _executed(state, "brief_answer"),
        },
    )


def planner_graph_node(
    state: RAGState,
    runtime: Runtime[GraphDependencies],
) -> dict[str, object]:
    """Graph adapter for the planner node."""
    return cast(
        "dict[str, object]",
        {
            **planner_node(state, config=get_config(), model=runtime.context.planner_model),
            "nodes_executed": _executed(state, "planner"),
        },
    )


def retrieve_graph_node(
    state: RAGState,
    runtime: Runtime[GraphDependencies],
) -> dict[str, object]:
    """Graph adapter for the retrieval node."""
    return cast(
        "dict[str, object]",
        {
            **retrieve_state_node(state, retriever=runtime.context.retriever),
            "nodes_executed": _executed(state, "retrieve"),
        },
    )


def rerank_graph_node(
    state: RAGState,
    runtime: Runtime[GraphDependencies],
) -> dict[str, object]:
    """Graph adapter for the rerank node."""
    return cast(
        "dict[str, object]",
        {
            **reranker_node(state, reranker=runtime.context.reranker),
            "nodes_executed": _executed(state, "rerank"),
        },
    )


def answer_graph_node(
    state: RAGState,
    runtime: Runtime[GraphDependencies],
) -> dict[str, object]:
    """Graph adapter for the grounded answer node."""
    return cast(
        "dict[str, object]",
        {
            **answerer_node(state, config=get_config(), model=runtime.context.answer_model),
            "nodes_executed": _executed(state, "answer"),
        },
    )


def review_graph_node(
    state: RAGState,
    runtime: Runtime[GraphDependencies],
) -> dict[str, object]:
    """Graph adapter for the groundedness review node."""
    result = reviewer_node(state, config=get_config(), model=runtime.context.reviewer_model)
    result["nodes_executed"] = _executed(state, "review")
    return result


def _next_node_from_state(state: RAGState) -> Literal["brief_answer", "planner"]:
    route = state.get("route")
    if not isinstance(route, RouteDecision):
        msg = "router state must include a route decision"
        raise TypeError(msg)
    return next_node_for_route(route)


def _next_node_after_review(state: RAGState) -> Literal["review", "finalize"]:
    review = state.get("review")
    if not isinstance(review, ReviewOutput):
        msg = "review state must include review output"
        raise TypeError(msg)
    if review.verdict is ReviewVerdict.REVISE:
        return "review"
    return "finalize"
