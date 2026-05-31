"""Answer generation helpers and LangGraph node for grounded CV responses."""

from __future__ import annotations

from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from cv_screener.rag.llm import build_structured_output_model, invoke_structured_output
from cv_screener.rag.prompts import ANSWERER_SYSTEM_PROMPT
from cv_screener.rag.schema import AnswerOutput

if TYPE_CHECKING:
    from langchain_core.language_models import LanguageModelInput
    from langchain_core.runnables import Runnable, RunnableConfig

    from cv_screener.config import RAGModelSettings
    from cv_screener.rag.state import RAGState
    from cv_screener.retrieval.schema import RetrievedChunk


ABSTAINED_ANSWER = "I don't have enough information in the indexed CVs to answer that."


def build_answer_model(
    settings: RAGModelSettings | None = None,
) -> Runnable[LanguageModelInput, AnswerOutput]:
    """Build the structured answer model for OpenAI-compatible chat backends."""
    return build_structured_output_model(AnswerOutput, settings=settings)


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

    return invoke_structured_output(
        _build_messages(user_query, chunks),
        model=model,
        schema=AnswerOutput,
        label="answerer",
        config=config,
    )


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
