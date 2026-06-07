"""Brief-response helpers and LangGraph node for non-retrieval turns."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from cv_screener.rag.llm import build_structured_output_model, invoke_structured_output
from cv_screener.rag.prompts import (
    BRIEF_ANSWER_SYSTEM_PROMPT,
    conversation_context_block,
)
from cv_screener.rag.schema import BriefAnswerOutput, RouteDecision, RouteTarget

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable, RunnableConfig

    from cv_screener.config import RAGConfig
    from cv_screener.rag.state import RAGState


def build_brief_answer_model(
    settings: RAGConfig,
) -> Runnable[LanguageModelInput, BriefAnswerOutput]:
    """Build the structured brief-answer model for OpenAI-compatible chat backends."""
    return build_structured_output_model(BriefAnswerOutput, settings=settings)


def answer_briefly(
    user_query: str,
    route: RouteDecision,
    *,
    model: Runnable[LanguageModelInput, BriefAnswerOutput],
    config: RunnableConfig | None = None,
    conversation_context: str | None = None,
) -> BriefAnswerOutput:
    """Respond briefly for small talk or clarification turns."""
    return invoke_structured_output(
        _build_messages(
            user_query,
            route,
            conversation_context=conversation_context,
        ),
        model=model,
        schema=BriefAnswerOutput,
        label="brief answerer",
        config=config,
    )


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

    conversation_context = state.get("conversation_context")
    if not isinstance(conversation_context, str):
        conversation_context = None
    return {
        "brief_answer": answer_briefly(
            user_query,
            route,
            model=model,
            config=config,
            conversation_context=conversation_context,
        )
    }


def _build_messages(
    user_query: str,
    route: RouteDecision,
    *,
    conversation_context: str | None = None,
) -> list[SystemMessage | HumanMessage]:
    context_block = conversation_context_block(conversation_context)
    human_content = "\n".join(
        [
            part
            for part in [
                context_block.rstrip(),
                f"Route: {route.route}",
                f"Reasoning: {route.reasoning}",
                f"User message: {user_query}",
            ]
            if part
        ]
    )
    return [
        SystemMessage(content=BRIEF_ANSWER_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]
