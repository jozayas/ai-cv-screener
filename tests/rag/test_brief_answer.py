from typing import cast

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import Runnable
from pydantic import SecretStr

import cv_screener.rag.llm as llm_module
from cv_screener.config import RAGConfig
from cv_screener.rag.nodes.brief_answer import (
    answer_briefly,
    brief_answer_node,
    build_brief_answer_model,
)
from cv_screener.rag.schema import BriefAnswerOutput, RouteDecision, RouteTarget


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


def test_build_brief_answer_model_parses_json_without_tool_calling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["init"] = kwargs

        def invoke(self, messages: object, config: object | None = None) -> object:
            del config
            captured["messages"] = messages
            return '{"text":"Hi."}'

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    result = answer_briefly(
        "hello",
        RouteDecision(route=RouteTarget.SMALL_TALK, reasoning="Greeting."),
        model=build_brief_answer_model(
            RAGConfig(
                openai_base_url="http://localhost:11434/v1",
                openai_api_key=SecretStr("ollama"),
                rag_model="gemma3:12b",
            )
        ),
    )

    init_kwargs = captured["init"]
    assert isinstance(init_kwargs, dict)
    assert result == BriefAnswerOutput(text="Hi.")
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert "Return only valid JSON. Do not call tools." in messages[-1].content
    assert "text" in messages[-1].content
    api_key = init_kwargs["api_key"]
    assert hasattr(api_key, "get_secret_value")
    assert api_key.get_secret_value() == "ollama"
