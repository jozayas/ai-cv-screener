from typing import cast

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import Runnable

import cv_screener.rag.brief_answer as brief_answer_module
from cv_screener.config import RAGModelSettings
from cv_screener.rag.brief_answer import (
    answer_briefly,
    brief_answer_node,
    build_brief_answer_model,
)
from cv_screener.rag.models import BriefAnswerOutput, RouteDecision, RouteTarget


def make_brief_runnable(
    response: object,
) -> tuple[Runnable[LanguageModelInput, BriefAnswerOutput], list[object]]:
    invocations: list[object] = []

    def invoke(messages: object) -> object:
        invocations.append(messages)
        return response

    return cast(
        "Runnable[LanguageModelInput, BriefAnswerOutput]", RunnableLambda(invoke)
    ), invocations


def test_brief_answer_handles_small_talk() -> None:
    model, invocations = make_brief_runnable({"text": "Hi. Ask me about the CVs."})

    result = answer_briefly(
        "hello",
        RouteDecision(route=RouteTarget.SMALL_TALK, reasoning="Greeting."),
        model=model,
    )

    assert result == BriefAnswerOutput(text="Hi. Ask me about the CVs.")
    assert len(invocations) == 1


def test_brief_answer_node_rejects_cv_query_route() -> None:
    model, _ = make_brief_runnable({"text": "unused"})

    with pytest.raises(ValueError, match="non-cv route decision"):
        brief_answer_node(
            {
                "user_query": "Who knows Python?",
                "route": RouteDecision(
                    route=RouteTarget.CV_QUERY,
                    reasoning="Needs retrieval.",
                ),
            },
            model=model,
        )


def test_build_brief_answer_model_uses_function_calling(
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
        ) -> Runnable[LanguageModelInput, BriefAnswerOutput]:
            captured["schema"] = schema
            captured["method"] = method
            runnable, _ = make_brief_runnable(BriefAnswerOutput(text="Hi."))
            return runnable

    monkeypatch.setattr(brief_answer_module, "ChatOpenAI", FakeChatOpenAI)
    result = answer_briefly(
        "hello",
        RouteDecision(route=RouteTarget.SMALL_TALK, reasoning="Greeting."),
        model=build_brief_answer_model(
            RAGModelSettings(
                openai_base_url="http://localhost:11434/v1",
                openai_api_key="ollama",
                rag_model="gemma3:12b",
            )
        ),
    )

    init_kwargs = captured["init"]
    assert isinstance(init_kwargs, dict)
    assert result == BriefAnswerOutput(text="Hi.")
    assert captured["schema"] is BriefAnswerOutput
    assert captured["method"] == "function_calling"
    api_key = init_kwargs["api_key"]
    assert hasattr(api_key, "get_secret_value")
    assert api_key.get_secret_value() == "ollama"
