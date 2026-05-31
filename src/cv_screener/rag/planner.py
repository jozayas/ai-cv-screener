"""Planner helpers and LangGraph node for retrieval-oriented query rewrites."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from cv_screener.config import RAGModelSettings
from cv_screener.rag.models import PlannerOutput, RouteDecision, RouteTarget
from cv_screener.rag.prompts import PLANNER_SYSTEM_PROMPT

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable, RunnableConfig

    from cv_screener.rag.state import RAGState


def build_planner_model(
    settings: RAGModelSettings | None = None,
) -> Runnable[LanguageModelInput, PlannerOutput]:
    """Build the structured planner model for OpenAI-compatible chat backends."""
    resolved_settings = settings or RAGModelSettings()
    llm = ChatOpenAI(
        model=resolved_settings.rag_model,
        base_url=resolved_settings.openai_base_url,
        api_key=resolved_settings.openai_api_key,
        temperature=resolved_settings.rag_temperature,
        max_retries=resolved_settings.rag_max_retries,
    )
    return cast(
        "Runnable[LanguageModelInput, PlannerOutput]",
        llm.with_structured_output(PlannerOutput, method="function_calling"),
    )


def plan_query(
    user_query: str,
    *,
    model: Runnable[LanguageModelInput, PlannerOutput],
    config: RunnableConfig | None = None,
) -> PlannerOutput:
    """Rewrite a recruiter-style query into retrieval-friendly search text."""
    result = model.invoke(_build_messages(user_query), config=config)
    try:
        return PlannerOutput.model_validate(result)
    except ValidationError as error:
        msg = f"planner returned invalid structured output: {error}"
        raise ValueError(msg) from error



def planner_node(
    state: RAGState,
    config: RunnableConfig | None = None,
    *,
    model: Runnable[LanguageModelInput, PlannerOutput],
) -> dict[str, PlannerOutput]:
    """LangGraph planner node that returns a state update.

    Bind ``model`` when wiring the graph, for example with ``functools.partial``.
    """
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
