"""Router helpers and LangGraph node for the RAG workflow."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from cv_screener.config import RAGModelSettings
from cv_screener.rag.models import RouteDecision, RouteTarget
from cv_screener.rag.prompts import ROUTER_SYSTEM_PROMPT

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable, RunnableConfig

    from cv_screener.rag.state import RAGState


def build_router_model(
    settings: RAGModelSettings | None = None,
) -> Runnable[LanguageModelInput, RouteDecision]:
    """Build the structured router model for OpenAI-compatible chat backends."""
    resolved_settings = settings or RAGModelSettings()
    llm = ChatOpenAI(
        model=resolved_settings.rag_model,
        base_url=resolved_settings.openai_base_url,
        api_key=resolved_settings.openai_api_key,
        temperature=resolved_settings.rag_temperature,
        max_retries=resolved_settings.rag_max_retries,
    )
    return cast(
        "Runnable[LanguageModelInput, RouteDecision]",
        llm.with_structured_output(RouteDecision, method="function_calling"),
    )


def route_query(
    user_query: str,
    *,
    model: Runnable[LanguageModelInput, RouteDecision],
    config: RunnableConfig | None = None,
) -> RouteDecision:
    """Classify a user query using structured output."""
    result = model.invoke(_build_messages(user_query), config=config)
    try:
        return RouteDecision.model_validate(result)
    except ValidationError as error:
        msg = f"router returned invalid structured output: {error}"
        raise ValueError(msg) from error


def router_node(
    state: RAGState,
    config: RunnableConfig | None = None,
    *,
    model: Runnable[LanguageModelInput, RouteDecision],
) -> dict[str, RouteDecision]:
    """LangGraph router node that returns a state update.

    Bind ``model`` when wiring the graph, for example with ``functools.partial``.
    """
    user_query = state.get("user_query")
    if not isinstance(user_query, str) or not user_query.strip():
        msg = "router state must include a non-empty user_query"
        raise ValueError(msg)
    return {"route": route_query(user_query, model=model, config=config)}


def next_node_for_route(
    decision: RouteDecision,
) -> Literal["planner", "finalize"]:
    """Map a router decision to the next graph node name."""
    if decision.route is RouteTarget.CV_QUERY:
        return "planner"
    return "finalize"


def _build_messages(user_query: str) -> list[SystemMessage | HumanMessage]:
    return [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=user_query),
    ]
