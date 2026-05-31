from typing import cast

import pytest
from langchain_core.language_models import LanguageModelInput
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables.base import Runnable
from pydantic import SecretStr

import cv_screener.rag.llm as llm_module
from cv_screener.config import RAGModelSettings
from cv_screener.rag.nodes.answer import (
    ABSTAINED_ANSWER,
    answer_query,
    answerer_node,
    build_answer_model,
)
from cv_screener.rag.schema import AnswerCitation, AnswerOutput
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
                AnswerCitation(rank=1, source_file="ada.pdf", page=1, section="Summary")
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
                    rank=1,
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
                    rank=1,
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


def test_answerer_build_model_parses_json_without_tool_calling(
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
                '{"answer":"Ada Lovelace has Python experience.",'
                '"citations":[{"rank":1,"source_file":"ada.pdf","page":2,"section":"Experience"}],'
                '"abstained":false}'
            )

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
    messages = captured["messages"]
    assert isinstance(messages, list)
    assert "Return only valid JSON. Do not call tools." in messages[-1].content
    assert "citations" in messages[-1].content
    api_key = init_kwargs["api_key"]
    assert hasattr(api_key, "get_secret_value")
    assert api_key.get_secret_value() == "ollama"


def test_answer_output_strips_citations_when_abstained() -> None:
    """Before-validator strips citations when abstained=True."""
    result = AnswerOutput(
        answer="No relevant candidates found.",
        citations=[AnswerCitation(rank=1, source_file="cv.pdf", page=1, section="Skills")],
        abstained=True,
    )
    assert result.abstained is True
    assert result.citations == []
    assert result.answer == "No relevant candidates found."


def test_answer_output_non_abstained_still_requires_citations() -> None:
    """After-validator still rejects non-abstained answers without citations."""
    with pytest.raises(ValueError, match="citations"):
        AnswerOutput(answer="Ada has Python experience.", abstained=False)


def test_answer_output_non_abstained_with_citations_works() -> None:
    """Normal non-abstained answers with citations still validate correctly."""
    result = AnswerOutput(
        answer="Ada has Python experience.",
        citations=[AnswerCitation(rank=1, source_file="ada.pdf", page=2, section="Skills")],
        abstained=False,
    )
    assert result.abstained is False
    assert len(result.citations) == 1
    assert result.citations[0].source_file == "ada.pdf"
