from typing import cast

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import Runnable
from pydantic import SecretStr

import cv_screener.rag.llm as llm_module
from cv_screener.config import RAGModelSettings
from cv_screener.rag.nodes.answer import ABSTAINED_ANSWER
from cv_screener.rag.nodes.review import (
    build_reviewer_model,
    review_answer,
    reviewer_node,
)
from cv_screener.rag.schema import (
    AnswerCitation,
    AnswerOutput,
    ReviewOutput,
    ReviewVerdict,
)
from cv_screener.retrieval.schema import RetrievedChunk


def make_reviewer_runnable(
    response: object,
) -> tuple[Runnable[LanguageModelInput, ReviewOutput], list[object]]:
    invocations: list[object] = []

    def invoke(messages: object) -> object:
        invocations.append(messages)
        return response

    return cast(
        "Runnable[LanguageModelInput, ReviewOutput]", RunnableLambda(invoke)
    ), invocations


def test_review_answer_skips_model_for_abstentions() -> None:
    model, invocations = make_reviewer_runnable(
        ReviewOutput(verdict=ReviewVerdict.ABSTAIN, reasoning="unused")
    )

    result = review_answer(
        "Who knows Rust?",
        AnswerOutput(answer=ABSTAINED_ANSWER, abstained=True),
        [],
        model=model,
    )

    assert result == ReviewOutput(
        verdict=ReviewVerdict.APPROVE,
        reasoning="The answer already abstains because evidence was insufficient.",
    )
    assert invocations == []


def test_reviewer_node_revises_answer_once() -> None:
    model, _ = make_reviewer_runnable(
        ReviewOutput(
            verdict=ReviewVerdict.REVISE,
            reasoning="The chunk supports Python APIs but not team leadership.",
            revised_answer="Ada Lovelace built Python APIs.",
        )
    )
    answer = AnswerOutput(
        answer="Ada Lovelace led the platform team and built Python APIs.",
        citations=[
            AnswerCitation(
                rank=1,
                source_file="ada.pdf",
                page=2,
                section="Experience",
            )
        ],
    )
    chunk = RetrievedChunk(
        source_file="ada.pdf",
        document_title="Ada Lovelace CV",
        page=2,
        section="Experience",
        text="Built Python APIs for internal platforms.",
        score=0.95,
        rank=1,
    )

    update = reviewer_node(
        {
            "user_query": "Who built Python APIs?",
            "answer": answer,
            "reranked_chunks": [chunk],
        },
        model=model,
    )

    assert update["review"] == ReviewOutput(
        verdict=ReviewVerdict.REVISE,
        reasoning="The chunk supports Python APIs but not team leadership.",
        revised_answer="Ada Lovelace built Python APIs.",
    )
    assert update["answer"] == answer.model_copy(
        update={"answer": "Ada Lovelace built Python APIs."}
    )
    assert update["review_attempts"] == 1


def test_reviewer_node_abstains_when_citation_does_not_match_chunks() -> None:
    model, invocations = make_reviewer_runnable(
        ReviewOutput(verdict=ReviewVerdict.APPROVE, reasoning="unused")
    )

    update = reviewer_node(
        {
            "user_query": "Who knows Python?",
            "answer": AnswerOutput(
                answer="Ada Lovelace has Python experience.",
                citations=[
                    AnswerCitation(
                        rank=1,
                        source_file="ada.pdf",
                        page=4,
                        section="Projects",
                    )
                ],
            ),
            "reranked_chunks": [
                RetrievedChunk(
                    source_file="ada.pdf",
                    document_title="Ada Lovelace CV",
                    page=2,
                    section="Experience",
                    text="Built Python APIs.",
                    score=0.9,
                    rank=1,
                )
            ],
        },
        model=model,
    )

    assert invocations == []
    assert update["review"] == ReviewOutput(
        verdict=ReviewVerdict.ABSTAIN,
        reasoning="No cited evidence was available to support the answer.",
    )
    assert update["answer"] == AnswerOutput(answer=ABSTAINED_ANSWER, abstained=True)


def test_reviewer_build_model_parses_json_without_tool_calling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: object) -> None:
            captured["init"] = kwargs

        def invoke(self, messages: object, config: object | None = None) -> object:
            del config
            captured["messages"] = messages
            return (
                '{"verdict":"approve",'
                '"reasoning":"The claim is fully supported by the cited chunk."}'
            )

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    result = review_answer(
        "Who has Python experience?",
        AnswerOutput(
            answer="Ada Lovelace has Python experience.",
            citations=[
                AnswerCitation(
                    rank=1, source_file="ada.pdf", page=2, section="Experience"
                )
            ],
        ),
        [
            RetrievedChunk(
                source_file="ada.pdf",
                document_title="Ada Lovelace CV",
                page=2,
                section="Experience",
                text="Built Python data pipelines.",
                score=0.8,
                rank=1,
            )
        ],
        model=build_reviewer_model(
            RAGModelSettings(
                openai_base_url="http://localhost:11434/v1",
                openai_api_key=SecretStr("ollama"),
                rag_model="gemma3:12b",
            )
        ),
    )

    init_kwargs = captured["init"]
    assert isinstance(init_kwargs, dict)
    assert result.verdict is ReviewVerdict.APPROVE
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert "Return only valid JSON. Do not call tools." in messages[-1].content
    assert "verdict" in messages[-1].content
    api_key = init_kwargs["api_key"]
    assert hasattr(api_key, "get_secret_value")
    assert api_key.get_secret_value() == "ollama"
