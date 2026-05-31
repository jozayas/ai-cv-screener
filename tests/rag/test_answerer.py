from typing import cast

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import Runnable
from pydantic import SecretStr

import cv_screener.rag.llm as llm_module
from cv_screener.config import RAGModelSettings
from cv_screener.rag.schema import AnswerCitation, AnswerOutput
from cv_screener.rag.nodes.answer import (
    ABSTAINED_ANSWER,
    answer_query,
    answerer_node,
    build_answer_model,
)
from cv_screener.retrieval.schema import RetrievedChunk


def make_answer_runnable(
    response: object,
) -> tuple[Runnable[LanguageModelInput, AnswerOutput], list[object]]:
    invocations: list[object] = []

    def invoke(messages: object) -> object:
        invocations.append(messages)
        return response

    return cast(
        "Runnable[LanguageModelInput, AnswerOutput]", RunnableLambda(invoke)
    ), invocations


def test_answer_query_abstains_without_chunks() -> None:
    model, invocations = make_answer_runnable(
        AnswerOutput(
            answer="unused",
            citations=[
                AnswerCitation(source_file="ada.pdf", page=1, section="Summary")
            ],
        )
    )

    result = answer_query("Who knows Python?", [], model=model)

    assert result == AnswerOutput(answer=ABSTAINED_ANSWER, abstained=True)
    assert invocations == []


def test_answerer_node_uses_reranked_chunks() -> None:
    model, _ = make_answer_runnable(
        AnswerOutput(
            answer="Ada Lovelace has Python backend experience.",
            citations=[
                AnswerCitation(
                    source_file="ada-lovelace.pdf",
                    page=2,
                    section="Experience",
                )
            ],
        )
    )
    update = answerer_node(
        {
            "user_query": "Who has Python backend experience?",
            "reranked_chunks": [
                RetrievedChunk(
                    source_file="ada-lovelace.pdf",
                    document_title="Ada Lovelace CV",
                    page=2,
                    section="Experience",
                    text="Built Python data pipelines and backend APIs.",
                    score=0.97,
                    rank=1,
                )
            ],
            "retrieved_chunks": [
                RetrievedChunk(
                    source_file="ignored.pdf",
                    document_title="Ignored",
                    page=1,
                    section="Summary",
                    text="Ignored chunk.",
                    score=0.1,
                    rank=1,
                )
            ],
        },
        model=model,
    )

    assert update == {
        "answer": AnswerOutput(
            answer="Ada Lovelace has Python backend experience.",
            citations=[
                AnswerCitation(
                    source_file="ada-lovelace.pdf",
                    page=2,
                    section="Experience",
                )
            ],
        )
    }


def test_answer_query_requires_citations_for_supported_answers() -> None:
    model, _ = make_answer_runnable(
        {"answer": "Ada has Python experience.", "abstained": False}
    )

    with pytest.raises(ValueError, match="citations"):
        answer_query(
            "Who knows Python?",
            [
                RetrievedChunk(
                    source_file="ada.pdf",
                    document_title="Ada Lovelace CV",
                    page=1,
                    section="Summary",
                    text="Python engineer.",
                    score=0.8,
                    rank=1,
                )
            ],
            model=model,
        )


def test_answerer_uses_function_calling_for_structured_output(
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
        ) -> Runnable[LanguageModelInput, AnswerOutput]:
            captured["schema"] = schema
            captured["method"] = method
            runnable, _ = make_answer_runnable(
                AnswerOutput(
                    answer="Ada Lovelace has Python experience.",
                    citations=[
                        AnswerCitation(
                            source_file="ada.pdf",
                            page=2,
                            section="Experience",
                        )
                    ],
                )
            )
            return runnable

    monkeypatch.setattr(llm_module, "ChatOpenAI", FakeChatOpenAI)
    result = answer_query(
        "Who has Python experience?",
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
        model=build_answer_model(
            RAGModelSettings(
                openai_base_url="http://localhost:11434/v1",
                openai_api_key=SecretStr("ollama"),
                rag_model="gemma3:12b",
            )
        ),
    )

    init_kwargs = captured["init"]
    assert isinstance(init_kwargs, dict)
    assert result.answer == "Ada Lovelace has Python experience."
    assert captured["schema"] is AnswerOutput
    assert captured["method"] == "function_calling"
    api_key = init_kwargs["api_key"]
    assert hasattr(api_key, "get_secret_value")
    assert api_key.get_secret_value() == "ollama"
