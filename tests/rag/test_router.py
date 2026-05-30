from typing import cast

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import Runnable

import cv_screener.rag.router as router_module
from cv_screener.config import RAGModelSettings
from cv_screener.rag.models import RouteDecision, RouteTarget
from cv_screener.rag.router import (
    build_router_model,
    next_node_for_route,
    route_query,
    router_node,
)


def make_router_runnable(
    response: object,
) -> tuple[Runnable[LanguageModelInput, RouteDecision], list[object]]:
    invocations: list[object] = []

    def invoke(messages: object) -> object:
        invocations.append(messages)
        return response

    return cast("Runnable[LanguageModelInput, RouteDecision]", RunnableLambda(invoke)), invocations


def test_router_parses_small_talk_and_bypasses_retrieval() -> None:
    model, invocations = make_router_runnable(
        {
            "route": "small_talk",
            "reasoning": "The user is greeting the assistant.",
            "small_talk_response": "Hi. Ask me about the CVs when you're ready.",
        }
    )
    decision = route_query("hi", model=model)

    assert decision == RouteDecision(
        route=RouteTarget.SMALL_TALK,
        reasoning="The user is greeting the assistant.",
        small_talk_response="Hi. Ask me about the CVs when you're ready.",
    )
    assert next_node_for_route(decision) == "finalize"
    assert len(invocations) == 1


def test_router_sends_cv_queries_to_planner() -> None:
    model, _ = make_router_runnable(
        RouteDecision(
            route=RouteTarget.CV_QUERY,
            reasoning="The user is asking about candidate experience.",
        )
    )
    decision = route_query("Who has Python experience?", model=model)

    assert decision.route is RouteTarget.CV_QUERY
    assert next_node_for_route(decision) == "planner"


def test_router_requires_clarification_text_for_clarification_route() -> None:
    model, _ = make_router_runnable(
        {
            "route": "needs_clarification",
            "reasoning": "The request is too broad to retrieve precisely.",
        }
    )
    with pytest.raises(ValueError, match="clarification_question"):
        route_query("Tell me about candidates.", model=model)


def test_router_node_returns_state_update() -> None:
    model, _ = make_router_runnable(
        RouteDecision(
            route=RouteTarget.CV_QUERY,
            reasoning="The user is asking a CV question.",
        )
    )
    update = router_node(
        {"user_query": "Which candidates know Python?"},
        model=model,
    )

    assert update == {
        "route": RouteDecision(
            route=RouteTarget.CV_QUERY,
            reasoning="The user is asking a CV question.",
        )
    }


def test_router_uses_function_calling_for_structured_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["init"] = kwargs

        def with_structured_output(
            self,
            schema: object,
            *,
            method: str,
        ) -> Runnable[LanguageModelInput, RouteDecision]:
            captured["schema"] = schema
            captured["method"] = method
            runnable, _ = make_router_runnable(
                RouteDecision(
                    route=RouteTarget.SMALL_TALK,
                    reasoning="Greeting detected.",
                    small_talk_response="Hi.",
                )
            )
            return runnable

    monkeypatch.setattr(router_module, "ChatOpenAI", FakeChatOpenAI)
    decision = route_query(
        "hello",
        model=build_router_model(
            RAGModelSettings(
                openai_base_url="http://localhost:11434/v1",
                openai_api_key="ollama",
                rag_model="gemma3:12b",
            )
        ),
    )

    init_kwargs = captured["init"]
    assert isinstance(init_kwargs, dict)
    assert decision.route is RouteTarget.SMALL_TALK
    assert captured["schema"] is RouteDecision
    assert captured["method"] == "function_calling"
    api_key = init_kwargs["api_key"]
    assert hasattr(api_key, "get_secret_value")
    assert api_key.get_secret_value() == "ollama"
