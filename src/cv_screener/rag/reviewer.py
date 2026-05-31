"""Groundedness review helpers and LangGraph node for CV answers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from cv_screener.config import RAGModelSettings
from cv_screener.rag.answerer import ABSTAINED_ANSWER
from cv_screener.rag.models import (
    AnswerCitation,
    AnswerOutput,
    ReviewOutput,
    ReviewVerdict,
)
from cv_screener.rag.prompts import REVIEWER_SYSTEM_PROMPT

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable, RunnableConfig

    from cv_screener.rag.state import RAGState
    from cv_screener.retrieval.schema import RetrievedChunk


DEFAULT_REVIEW_PASSES = 1


def build_reviewer_model(
    settings: RAGModelSettings | None = None,
) -> Runnable[LanguageModelInput, ReviewOutput]:
    """Build the structured reviewer model for OpenAI-compatible chat backends."""
    resolved_settings = settings or RAGModelSettings()
    llm = ChatOpenAI(
        model=resolved_settings.rag_model,
        base_url=resolved_settings.openai_base_url,
        api_key=resolved_settings.openai_api_key,
        temperature=resolved_settings.rag_temperature,
        max_retries=resolved_settings.rag_max_retries,
    )
    return cast(
        "Runnable[LanguageModelInput, ReviewOutput]",
        llm.with_structured_output(ReviewOutput, method="function_calling"),
    )


def review_answer(
    user_query: str,
    answer: AnswerOutput,
    chunks: list[RetrievedChunk],
    *,
    model: Runnable[LanguageModelInput, ReviewOutput],
    config: RunnableConfig | None = None,
) -> ReviewOutput:
    """Review whether an answer is grounded in the cited CV evidence."""
    if answer.abstained:
        return ReviewOutput(
            verdict=ReviewVerdict.APPROVE,
            reasoning="The answer already abstains because evidence was insufficient.",
        )
    if not chunks:
        return ReviewOutput(
            verdict=ReviewVerdict.ABSTAIN,
            reasoning="No cited evidence was available to support the answer.",
        )

    result = model.invoke(_build_messages(user_query, answer, chunks), config=config)
    try:
        return ReviewOutput.model_validate(result)
    except ValidationError as error:
        msg = f"reviewer returned invalid structured output: {error}"
        raise ValueError(msg) from error


def reviewer_node(
    state: RAGState,
    config: RunnableConfig | None = None,
    *,
    model: Runnable[LanguageModelInput, ReviewOutput],
    max_review_passes: int = DEFAULT_REVIEW_PASSES,
) -> dict[str, object]:
    """LangGraph reviewer node that can approve, revise, or abstain."""
    user_query = state.get("user_query")
    if not isinstance(user_query, str) or not user_query.strip():
        msg = "review state must include a non-empty user_query"
        raise ValueError(msg)

    answer = state.get("answer")
    if not isinstance(answer, AnswerOutput):
        msg = "review state must include answer output"
        raise TypeError(msg)

    available_chunks = state.get("reranked_chunks")
    if available_chunks is None:
        available_chunks = state.get("retrieved_chunks", [])

    cited_chunks = _match_citations(answer.citations, available_chunks)
    review = review_answer(
        user_query,
        answer,
        cited_chunks,
        model=model,
        config=config,
    )

    update: dict[str, object] = {"review": review}
    review_attempts = _coerce_review_attempts(state.get("review_attempts"))
    if review.verdict is ReviewVerdict.REVISE:
        if review_attempts >= max_review_passes:
            update["answer"] = AnswerOutput(answer=ABSTAINED_ANSWER, abstained=True)
            update["review"] = ReviewOutput(
                verdict=ReviewVerdict.ABSTAIN,
                reasoning="The answer could not be grounded within the review pass limit.",
            )
            return update

        update["answer"] = answer.model_copy(update={"answer": review.revised_answer})
        update["review_attempts"] = review_attempts + 1
        return update

    if review.verdict is ReviewVerdict.ABSTAIN:
        update["answer"] = AnswerOutput(answer=ABSTAINED_ANSWER, abstained=True)

    return update


def _build_messages(
    user_query: str,
    answer: AnswerOutput,
    chunks: list[RetrievedChunk],
) -> list[SystemMessage | HumanMessage]:
    return [
        SystemMessage(content=REVIEWER_SYSTEM_PROMPT),
        HumanMessage(content=_build_user_prompt(user_query, answer, chunks)),
    ]


def _build_user_prompt(
    user_query: str,
    answer: AnswerOutput,
    chunks: list[RetrievedChunk],
) -> str:
    citation_lines = [
        f"- {citation.source_file} | page {citation.page} | {citation.section}"
        for citation in answer.citations
    ]
    evidence_blocks = []
    for index, chunk in enumerate(chunks, start=1):
        evidence_blocks.append(
            "\n".join(
                [
                    f"Evidence {index}",
                    f"source_file: {chunk.source_file}",
                    f"page: {chunk.page}",
                    f"section: {chunk.section}",
                    f"text: {chunk.text}",
                ]
            )
        )
    return "\n\n".join(
        [
            f"User question: {user_query}",
            f"Answer draft: {answer.answer}",
            "Citations:",
            *citation_lines,
            "Evidence:",
            *evidence_blocks,
        ]
    )


def _match_citations(
    citations: list[AnswerCitation],
    chunks: list[RetrievedChunk],
) -> list[RetrievedChunk]:
    chunk_by_key = {
        (chunk.source_file, chunk.page, chunk.section): chunk for chunk in chunks
    }
    matched_chunks: list[RetrievedChunk] = []
    for citation in citations:
        chunk = chunk_by_key.get(
            (citation.source_file, citation.page, citation.section)
        )
        if chunk is None:
            return []
        matched_chunks.append(chunk)
    return matched_chunks


def _coerce_review_attempts(value: object) -> int:
    if isinstance(value, int) and value >= 0:
        return value
    return 0
