from typing import cast

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import Runnable
from pydantic import SecretStr

import cv_screener.rag.llm as llm_module
from cv_screener.config import RAGConfig
from cv_screener.rag.nodes.route import (
    build_router_model,
    next_node_for_route,
    route_query,
    router_node,
)
from cv_screener.rag.prompts import ROUTER_SYSTEM_PROMPT
from cv_screener.rag.schema import RouteDecision, RouteTarget


def make_router_runnable(
    response: object,
) -> tuple[Runnable[LanguageModelInput, RouteDecision], list[object]]:
    invocations: list[object] = []

    def invoke(messages: object) -> object:
        invocations.append(messages)
        return response

    return cast(
        "Runnable[LanguageModelInput, RouteDecision]", RunnableLambda(invoke)
    ), invocations


def test_router_parses_small_talk_and_bypasses_retrieval() -> None:
    model, invocations = make_router_runnable(
        {
            "route": "small_talk",
            "reasoning": "The user is greeting the assistant.",
        }
    )
    decision = route_query("hi", model=model)

    assert decision == RouteDecision(
        route=RouteTarget.SMALL_TALK,
        reasoning="The user is greeting the assistant.",
    )
    assert next_node_for_route(decision) == "brief_answer"
    assert len(invocations) == 1


def test_router_sends_cv_queries_to_planner() -> None:
    model, _ = make_router_runnable(
        RouteDecision(
            route=RouteTarget.CV_QUERY,
            reasoning="The user is asking about candidate experience.",
        )
    )
    decision = route_query("Which candidates led platform migrations?", model=model)

    assert decision.route is RouteTarget.CV_QUERY
    assert next_node_for_route(decision) == "planner"


def test_router_node_uses_structured_model_for_cv_queries() -> None:
    model, invocations = make_router_runnable(
        RouteDecision(
            route=RouteTarget.CV_QUERY,
            reasoning="The user asks about candidate experience.",
        )
    )

    update = router_node(
        {"user_query": "Which candidates led platform migrations?"},
        model=model,
    )

    route = update["route"]
    assert route.route is RouteTarget.CV_QUERY
    assert next_node_for_route(route) == "planner"
    assert len(invocations) == 1


def test_router_sends_python_experience_candidate_queries_to_targeted_lookup() -> None:
    update = router_node(
        {"user_query": "Who has Python experience?"},
        model=make_router_runnable(
            RouteDecision(
                route=RouteTarget.TARGETED_LOOKUP,
                reasoning="The user asks for an exact skill lookup.",
            )
        )[0],
    )

    route = update["route"]
    assert route.route is RouteTarget.TARGETED_LOOKUP
    assert next_node_for_route(route) == "targeted_lookup"


def test_router_sends_experience_with_skill_queries_to_targeted_lookup() -> None:
    update = router_node(
        {"user_query": "Who has experience with Python?"},
        model=make_router_runnable(
            RouteDecision(
                route=RouteTarget.TARGETED_LOOKUP,
                reasoning="The user asks for an exact skill lookup.",
            )
        )[0],
    )

    route = update["route"]
    assert route.route is RouteTarget.TARGETED_LOOKUP
    assert next_node_for_route(route) == "targeted_lookup"


def test_router_prompt_classifies_exact_skill_queries_as_targeted_lookup() -> None:
    assert "Who knows <skill>?" in ROUTER_SYSTEM_PROMPT
    assert "Who has <skill> experience?" in ROUTER_SYSTEM_PROMPT
    assert "targeted_lookup" in ROUTER_SYSTEM_PROMPT


def test_router_accepts_clarification_route_without_extra_payload() -> None:
    model, _ = make_router_runnable(
        {
            "route": "needs_clarification",
            "reasoning": "The request is too broad to retrieve precisely.",
        }
    )
    decision = route_query("Tell me about candidates.", model=model)

    assert decision.route is RouteTarget.NEEDS_CLARIFICATION
    assert next_node_for_route(decision) == "brief_answer"


def test_router_node_returns_state_update() -> None:
    model, invocations = make_router_runnable(
        RouteDecision(
            route=RouteTarget.CV_QUERY,
            reasoning="The user asks about candidate skills.",
        )
    )
    update = router_node(
        {"user_query": "Which candidates know Python?"},
        model=model,
    )

    route = update["route"]
    assert route.route is RouteTarget.CV_QUERY
    assert len(invocations) == 1


def test_router_build_model_parses_json_without_tool_calling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["init"] = kwargs

        def invoke(self, messages: object, config: object | None = None) -> object:
            del config
            captured["messages"] = messages
            return '{"route":"small_talk","reasoning":"Greeting detected."}'

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    decision = route_query(
        "hello",
        model=build_router_model(
            RAGConfig(
                openai_base_url="http://localhost:11434/v1",
                openai_api_key=SecretStr("ollama"),
                rag_model="gemma3:12b",
            )
        ),
    )

    init_kwargs = captured["init"]
    assert isinstance(init_kwargs, dict)
    assert decision.route is RouteTarget.SMALL_TALK
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert "Return only valid JSON. Do not call tools." in messages[-1].content
    assert "route" in messages[-1].content
    api_key = init_kwargs["api_key"]
    assert hasattr(api_key, "get_secret_value")
    assert api_key.get_secret_value() == "ollama"
