"""Brief-response helpers and LangGraph node for non-retrieval turns."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from cv_screener.config import RAGModelSettings
from cv_screener.rag.models import BriefAnswerOutput, RouteDecision, RouteTarget
from cv_screener.rag.prompts import BRIEF_ANSWER_SYSTEM_PROMPT

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable, RunnableConfig

    from cv_screener.rag.state import RAGState


def build_brief_answer_model(
    settings: RAGModelSettings | None = None,
) -> Runnable[LanguageModelInput, BriefAnswerOutput]:
    """Build the structured brief-answer model for OpenAI-compatible chat backends."""
    resolved_settings = settings or RAGModelSettings()
    llm = ChatOpenAI(
        model=resolved_settings.rag_model,
        base_url=resolved_settings.openai_base_url,
        api_key=resolved_settings.openai_api_key,
        temperature=resolved_settings.rag_temperature,
        max_retries=resolved_settings.rag_max_retries,
    )
    return cast(
        "Runnable[LanguageModelInput, BriefAnswerOutput]",
        llm.with_structured_output(BriefAnswerOutput, method="function_calling"),
    )


def answer_briefly(
    user_query: str,
    route: RouteDecision,
    *,
    model: Runnable[LanguageModelInput, BriefAnswerOutput],
    config: RunnableConfig | None = None,
) -> BriefAnswerOutput:
    """Respond briefly for small talk or clarification turns."""
    result = model.invoke(_build_messages(user_query, route), config=config)
    try:
        return BriefAnswerOutput.model_validate(result)
    except ValidationError as error:
        msg = f"brief answerer returned invalid structured output: {error}"
        raise ValueError(msg) from error


def brief_answer_node(
    state: RAGState,
    config: RunnableConfig | None = None,
    *,
    model: Runnable[LanguageModelInput, BriefAnswerOutput],
) -> dict[str, BriefAnswerOutput]:
    """LangGraph brief-answer node for non-retrieval turns."""
    user_query = state.get("user_query")
    if not isinstance(user_query, str) or not user_query.strip():
        msg = "brief answer state must include a non-empty user_query"
        raise ValueError(msg)

    route = state.get("route")
    if not isinstance(route, RouteDecision) or route.route is RouteTarget.CV_QUERY:
        msg = "brief answer state must include a non-cv route decision"
        raise ValueError(msg)

    return {"brief_answer": answer_briefly(user_query, route, model=model, config=config)}


def _build_messages(
    user_query: str,
    route: RouteDecision,
) -> list[SystemMessage | HumanMessage]:
    return [
        SystemMessage(content=BRIEF_ANSWER_SYSTEM_PROMPT),
        HumanMessage(
            content="\n".join(
                [
                    f"Route: {route.route}",
                    f"Reasoning: {route.reasoning}",
                    f"User message: {user_query}",
                ]
            )
        ),
    ]
