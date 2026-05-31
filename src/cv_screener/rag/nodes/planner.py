"""Planner helpers and LangGraph node for retrieval-oriented query rewrites."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from cv_screener.rag.llm import build_structured_output_model, invoke_structured_output
from cv_screener.rag.schema import PlannerOutput, RouteDecision, RouteTarget
from cv_screener.rag.prompts import PLANNER_SYSTEM_PROMPT

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable, RunnableConfig

    from cv_screener.config import RAGModelSettings
    from cv_screener.rag.state import RAGState


def build_planner_model(
    settings: RAGModelSettings | None = None,
) -> Runnable[LanguageModelInput, PlannerOutput]:
    """Build the structured planner model for OpenAI-compatible chat backends."""
    return build_structured_output_model(PlannerOutput, settings=settings)


def plan_query(
    user_query: str,
    *,
    model: Runnable[LanguageModelInput, PlannerOutput],
    config: RunnableConfig | None = None,
) -> PlannerOutput:
    """Rewrite a recruiter-style query into retrieval-friendly search text."""
    return invoke_structured_output(
        _build_messages(user_query),
        model=model,
        schema=PlannerOutput,
        label="planner",
        config=config,
    )


def planner_node(
    state: RAGState,
    config: RunnableConfig | None = None,
    *,
    model: Runnable[LanguageModelInput, PlannerOutput],
) -> dict[str, PlannerOutput]:
    """LangGraph planner node that returns a state update."""
    user_query = state.get("user_query")
    if not isinstance(user_query, str) or not user_query.strip():
        msg = "planner state must include a non-empty user_query"
        raise ValueError(msg)

    route = state.get("route")
    if not isinstance(route, RouteDecision) or route.route is not RouteTarget.CV_QUERY:
        msg = "planner state must include route=cv_query"
        raise ValueError(msg)

    return {"planner": plan_query(user_query, model=model, config=config)}


def _build_messages(user_query: str) -> list[SystemMessage | HumanMessage]:
    return [
        SystemMessage(content=PLANNER_SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ]
