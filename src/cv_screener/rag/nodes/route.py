"""Router helpers and LangGraph node for the RAG workflow."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Literal

from langchain_core.messages import HumanMessage, SystemMessage

from cv_screener.rag.llm import build_structured_output_model, invoke_structured_output
from cv_screener.rag.prompts import (
    ROUTER_SYSTEM_PROMPT,
    conversation_context_block,
)
from cv_screener.rag.schema import RouteDecision, RouteTarget

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable, RunnableConfig

    from cv_screener.config import RAGModelSettings
    from cv_screener.rag.state import RAGState


def build_router_model(
    settings: RAGModelSettings | None = None,
) -> Runnable[LanguageModelInput, RouteDecision]:
    """Build the structured router model for OpenAI-compatible chat backends."""
    return build_structured_output_model(RouteDecision, settings=settings)


def route_query(
    user_query: str,
    *,
    model: Runnable[LanguageModelInput, RouteDecision],
    config: RunnableConfig | None = None,
    conversation_context: str | None = None,
) -> RouteDecision:
    """Classify a user query using structured output."""
    return invoke_structured_output(
        _build_messages(user_query, conversation_context=conversation_context),
        model=model,
        schema=RouteDecision,
        label="router",
        config=config,
    )


def router_node(
    state: RAGState,
    config: RunnableConfig | None = None,
    *,
    model: Runnable[LanguageModelInput, RouteDecision],
) -> dict[str, RouteDecision]:
    """LangGraph router node that returns a state update."""
    user_query = state.get("user_query")
    if not isinstance(user_query, str) or not user_query.strip():
        msg = "router state must include a non-empty user_query"
        raise ValueError(msg)
    deterministic = _deterministic_route(user_query)
    if deterministic is not None:
        return {"route": deterministic}
    conversation_context = state.get("conversation_context")
    if not isinstance(conversation_context, str):
        conversation_context = None
    return {
        "route": route_query(
            user_query,
            model=model,
            config=config,
            conversation_context=conversation_context,
        )
    }


def next_node_for_route(
    decision: RouteDecision,
) -> Literal["planner", "brief_answer", "targeted_lookup", "return_cv"]:
    """Map a router decision to the next graph node name."""
    if decision.route is RouteTarget.FULL_CV:
        return "return_cv"
    if decision.route is RouteTarget.TARGETED_LOOKUP:
        return "targeted_lookup"
    if decision.route is RouteTarget.CV_QUERY:
        return "planner"
    return "brief_answer"


def _deterministic_route(user_query: str) -> RouteDecision | None:
    text = user_query.strip()
    lowered = text.casefold()
    if _is_full_cv_query(lowered):
        return RouteDecision(
            route=RouteTarget.FULL_CV,
            reasoning="The query explicitly asks for a candidate CV document.",
        )
    if _is_targeted_lookup_query(lowered):
        return RouteDecision(
            route=RouteTarget.TARGETED_LOOKUP,
            reasoning="The query is a deterministic candidate or entity lookup.",
        )
    return None


def _is_full_cv_query(lowered: str) -> bool:
    return bool(
        re.search(r"\b(?:cv|resume)\s+(?:of|for)\b", lowered)
        or re.search(r"\bgive me (?:the )?(?:cv|resume)\b", lowered)
    )


def _is_targeted_lookup_query(lowered: str) -> bool:
    if re.search(r"\bgraduated from\b", lowered):
        return True
    if re.search(r"\bprofile of\b", lowered):
        return True
    if re.search(r"\bsummarize\b", lowered) and "experience" not in lowered:
        return True
    skill_match = re.search(
        r"\bwho (?:has experience with\s+(.+?)|(?:has|knows)\s+(.+?))[?.!]*$",
        lowered,
    )
    if skill_match is None:
        return False
    skill_tail = next(group for group in skill_match.groups() if group is not None)
    has_simple_experience_skill = bool(
        re.fullmatch(r"[a-z0-9+#./-]+\s+experience", skill_tail)
    )
    if "experience" in skill_tail:
        return has_simple_experience_skill
    blocked_terms = ("background", "worked", "leadership")
    return not any(term in skill_tail for term in blocked_terms)


def _build_messages(
    user_query: str,
    *,
    conversation_context: str | None = None,
) -> list[SystemMessage | HumanMessage]:
    context_block = conversation_context_block(conversation_context)
    human_content = "\n".join(
        [
            part
            for part in [context_block.rstrip(), f"User message: {user_query}"]
            if part
        ]
    )
    return [
        SystemMessage(content=ROUTER_SYSTEM_PROMPT),
        HumanMessage(content=human_content),
    ]
