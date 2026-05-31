"""Answer generation helpers and LangGraph node for grounded CV responses."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from cv_screener.config import RAGModelSettings
from cv_screener.rag.prompts import ANSWERER_SYSTEM_PROMPT
from cv_screener.rag.schema import AnswerOutput

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable, RunnableConfig

    from cv_screener.rag.state import RAGState
    from cv_screener.retrieval.schema import RetrievedChunk


ABSTAINED_ANSWER = "I don't have enough information in the indexed CVs to answer that."


def build_answer_model(
    settings: RAGModelSettings | None = None,
) -> Runnable[LanguageModelInput, AnswerOutput]:
    """Build the structured answer model for OpenAI-compatible chat backends."""
    resolved_settings = settings or RAGModelSettings()
    llm = ChatOpenAI(
        model=resolved_settings.rag_model,
        base_url=resolved_settings.openai_base_url,
        api_key=resolved_settings.openai_api_key,
        temperature=resolved_settings.rag_temperature,
        max_retries=resolved_settings.rag_max_retries,
    )
    return cast(
        "Runnable[LanguageModelInput, AnswerOutput]",
        llm.with_structured_output(AnswerOutput, method="function_calling"),
    )


def answer_query(
    user_query: str,
    chunks: list[RetrievedChunk],
    *,
    model: Runnable[LanguageModelInput, AnswerOutput],
    config: RunnableConfig | None = None,
) -> AnswerOutput:
    """Answer a recruiter-style question using only retrieved CV evidence."""
    if not chunks:
        return AnswerOutput(answer=ABSTAINED_ANSWER, abstained=True)

    result = model.invoke(_build_messages(user_query, chunks), config=config)
    try:
        return AnswerOutput.model_validate(result)
    except ValidationError as error:
        msg = f"answerer returned invalid structured output: {error}"
        raise ValueError(msg) from error


def answerer_node(
    state: RAGState,
    config: RunnableConfig | None = None,
    *,
    model: Runnable[LanguageModelInput, AnswerOutput],
) -> dict[str, AnswerOutput]:
    """LangGraph answerer node that returns a state update."""
    user_query = state.get("user_query")
    if not isinstance(user_query, str) or not user_query.strip():
        msg = "answer state must include a non-empty user_query"
        raise ValueError(msg)

    chunks = state.get("reranked_chunks")
    if chunks is None:
        chunks = state.get("retrieved_chunks", [])

    return {"answer": answer_query(user_query, chunks, model=model, config=config)}


def _build_messages(
    user_query: str,
    chunks: list[RetrievedChunk],
) -> list[SystemMessage | HumanMessage]:
    return [
        SystemMessage(content=ANSWERER_SYSTEM_PROMPT),
        HumanMessage(content=_build_user_prompt(user_query, chunks)),
    ]


def _build_user_prompt(user_query: str, chunks: list[RetrievedChunk]) -> str:
    context_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"Chunk {index}",
                    f"source_file: {chunk.source_file}",
                    f"page: {chunk.page}",
                    f"section: {chunk.section}",
                    f"text: {chunk.text}",
                ]
            )
        )
    return "\n\n".join([f"User question: {user_query}", "Context:", *context_blocks])
