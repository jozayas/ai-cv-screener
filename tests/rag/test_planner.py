from typing import cast

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import Runnable
from pydantic import SecretStr

import cv_screener.rag.llm as llm_module
from cv_screener.config import RAGModelSettings
from cv_screener.rag.schema import (
    PlannerOutput,
    RouteDecision,
    RouteTarget,
    SearchFacets,
)
from cv_screener.rag.nodes.planner import (
    build_planner_model,
    plan_query,
    planner_node,
)


def make_planner_runnable(
    response: object,
) -> tuple[Runnable[LanguageModelInput, PlannerOutput], list[object]]:
    invocations: list[object] = []

    def invoke(messages: object) -> object:
        invocations.append(messages)
        return response

    return cast(
        "Runnable[LanguageModelInput, PlannerOutput]", RunnableLambda(invoke)
    ), invocations


def test_planner_normalizes_recruiter_query_for_retrieval() -> None:
    model, invocations = make_planner_runnable(
        {
            "primary_query": "backend engineer python aws",
            "alternate_queries": ["python developer aws", "backend engineer cloud"],
            "facets": {
                "skills": ["Python", "AWS"],
                "roles": ["Backend Engineer"],
                "seniority": "mid",
            },
        }
    )

    result = plan_query(
        "Find mid-level backend engineers with Python and AWS experience.",
        model=model,
    )

    assert result == PlannerOutput(
        primary_query="backend engineer python aws",
        alternate_queries=["python developer aws", "backend engineer cloud"],
        facets=SearchFacets(
            skills=["Python", "AWS"],
            roles=["Backend Engineer"],
            seniority="mid",
        ),
    )
    assert len(invocations) == 1


def test_planner_node_returns_state_update() -> None:
    model, _ = make_planner_runnable(
        PlannerOutput(
            primary_query="machine learning engineer python",
            alternate_queries=["ml engineer python"],
        )
    )

    update = planner_node(
        {
            "user_query": "Which candidates are machine learning engineers with Python?",
            "route": RouteDecision(
                route=RouteTarget.CV_QUERY,
                reasoning="The user is asking about candidate qualifications.",
            ),
        },
        model=model,
    )

    assert update == {
        "planner": PlannerOutput(
            primary_query="machine learning engineer python",
            alternate_queries=["ml engineer python"],
        )
    }


def test_planner_requires_primary_query() -> None:
    model, _ = make_planner_runnable(
        {
            "alternate_queries": ["python engineer"],
            "facets": {},
        }
    )

    with pytest.raises(ValueError, match="primary_query"):
        plan_query("Who knows Python?", model=model)


def test_planner_uses_function_calling_for_structured_output(
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
        ) -> Runnable[LanguageModelInput, PlannerOutput]:
            captured["schema"] = schema
            captured["method"] = method
            runnable, _ = make_planner_runnable(
                PlannerOutput(primary_query="python engineer")
            )
            return runnable

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    result = plan_query(
        "Who has Python experience?",
        model=build_planner_model(
            RAGModelSettings(
                openai_base_url="http://localhost:11434/v1",
                openai_api_key=SecretStr("ollama"),
                rag_model="gemma3:12b",
            )
        ),
    )

    init_kwargs = captured["init"]
    assert isinstance(init_kwargs, dict)
    assert result.primary_query == "python engineer"
    assert captured["schema"] is PlannerOutput
    assert captured["method"] == "function_calling"
    api_key = init_kwargs["api_key"]
    assert hasattr(api_key, "get_secret_value")
    assert api_key.get_secret_value() == "ollama"
