import pytest
from langchain_core.messages import AIMessage
from pydantic import SecretStr

import cv_screener.rag.llm as llm_module
from cv_screener.config import RAGConfig
from cv_screener.rag.llm import build_structured_output_model
from cv_screener.rag.schema import RouteDecision, RouteTarget


def test_build_structured_output_model_repairs_invalid_json_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {"calls": []}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["init"] = kwargs

        def invoke(self, messages: object, config: object | None = None) -> object:
            del config
            calls = captured["calls"]
            assert isinstance(calls, list)
            calls.append(messages)
            if len(calls) == 1:
                return "small_talk"
            return '{"route":"small_talk","reasoning":"Greeting detected."}'

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    model = build_structured_output_model(
        RouteDecision,
        settings=RAGConfig(
            openai_base_url="http://localhost:11434/v1",
            openai_api_key=SecretStr("ollama"),
            rag_model="gemma3:12b",
        ),
    )

    result = model.invoke("hello")

    assert result == RouteDecision(
        route=RouteTarget.SMALL_TALK,
        reasoning="Greeting detected.",
    )
    calls = captured["calls"]
    assert isinstance(calls, list)
    assert len(calls) == 2


def test_build_structured_output_model_extracts_json_from_content_dict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def invoke(self, messages: object, config: object | None = None) -> object:
            del messages, config
            return {
                "content": '{\n  "route": "small_talk",\n  "reasoning": "Greeting detected."\n}',
                "additional_kwargs": {},
                "response_metadata": {
                    "token_usage": {"prompt_tokens": 0, "completion_tokens": 0}
                },
            }

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    model = build_structured_output_model(
        RouteDecision,
        settings=RAGConfig(
            openai_base_url="http://localhost:11434/v1",
            openai_api_key=SecretStr("ollama"),
            rag_model="gemma3:12b",
        ),
    )

    result = model.invoke("hello")

    assert result == RouteDecision(
        route=RouteTarget.SMALL_TALK,
        reasoning="Greeting detected.",
    )


def test_build_structured_output_model_extracts_json_from_ai_message(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            del kwargs

        def invoke(self, messages: object, config: object | None = None) -> object:
            del messages, config
            return AIMessage(
                content='{"route":"small_talk","reasoning":"Greeting detected."}'
            )

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    model = build_structured_output_model(
        RouteDecision,
        settings=RAGConfig(
            openai_base_url="http://localhost:11434/v1",
            openai_api_key=SecretStr("ollama"),
            rag_model="gemma3:12b",
        ),
    )

    result = model.invoke("hello")

    assert result == RouteDecision(
        route=RouteTarget.SMALL_TALK,
        reasoning="Greeting detected.",
    )
