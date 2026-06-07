"""Chainlit chat UI for the CV screener."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import chainlit as cl
from chainlit.config import config

from cv_screener.chainlit.ui import (
    EMPTY_QUERY_MESSAGE,
    RUNTIME_ERROR_MESSAGE,
    candidate_reference_clarification,
    candidate_references_from_answer,
    rewrite_query_with_candidate_reference,
)
from cv_screener.cli.dependencies import (
    RAGQueryServiceProtocol,
    build_rag_query_service,
)
from cv_screener.config import AppSettings

config.ui.cot = "tool_call"

NODE_LABELS: dict[str, str] = {
    "router": "Router",
    "brief_answer": "Quick Answer",
    "planner": "Search Planner",
    "retrieve": "CV Retriever",
    "rerank": "Result Reranker",
    "answer": "Answer Generator",
    "review": "Answer Reviewer",
}


def _step_summary(node_name: str, state_update: dict[str, Any]) -> str:
    if node_name == "planner":
        planner = state_update.get("planner")
        return f"query: {planner.primary_query}" if planner else ""
    if node_name == "retrieve":
        chunks = state_update.get("retrieved_chunks", [])
        return f"{len(chunks)} chunks found"
    if node_name == "rerank":
        chunks = state_update.get("reranked_chunks", [])
        return f"top {len(chunks)} results"
    if node_name == "answer":
        answer = state_update.get("answer")
        return "generated" if answer else ""
    if node_name == "review":
        review = state_update.get("review")
        return review.verdict if review else ""
    return ""


def _get_query_service() -> RAGQueryServiceProtocol:
    """Reuse one query service instance per chat session."""
    service = cast(
        "RAGQueryServiceProtocol | None",
        cl.user_session.get("rag_query_service"),
    )
    if service is None:
        settings = AppSettings()
        service = build_rag_query_service(
            rag_settings=settings.rag,
            sqlite_path=Path(settings.sqlite.sqlite_path),
            candidate_name_min_score=settings.lookup.candidate_name_min_score,
            qdrant_settings=settings.qdrant,
        )
        cl.user_session.set("rag_query_service", service)
    return service


def _get_candidate_refs() -> list[str]:
    refs = cast("list[str] | None", cl.user_session.get("candidate_refs"))
    return refs or []


def _get_conversation_context() -> str:
    context = cast("str | None", cl.user_session.get("conversation_context"))
    return context or ""


def _set_conversation_context(*, user_query: str, state_update: dict[str, Any]) -> None:
    candidate_refs = _candidate_refs_from_state_update(state_update)
    full_cv = state_update.get("full_cv")
    final_text = cast("str | None", state_update.get("final_text"))
    parts: list[str] = [f"Last user question: {user_query}"]
    if candidate_refs:
        parts.append("Last candidates: " + ", ".join(candidate_refs))
    if full_cv is not None:
        candidate_name = getattr(full_cv, "candidate_name", None)
        source_file = getattr(full_cv, "source_file", None)
        if candidate_name:
            parts.append(f"Last CV candidate: {candidate_name}")
        if source_file:
            parts.append(f"Last document: {source_file}")
    elif final_text:
        parts.append("Last assistant response: " + final_text[:300])
    cl.user_session.set("conversation_context", "\n".join(parts))


def _candidate_refs_from_state_update(state_update: dict[str, Any]) -> list[str]:
    full_cv = state_update.get("full_cv")
    candidate_name = getattr(full_cv, "candidate_name", None)
    if candidate_name:
        return [str(candidate_name)]

    targeted_lookup = state_update.get("targeted_lookup")
    candidate_names = getattr(targeted_lookup, "candidate_names", None)
    if isinstance(candidate_names, list) and candidate_names:
        return [str(name) for name in candidate_names if str(name).strip()]

    answer = state_update.get("answer")
    if answer is not None:
        return candidate_references_from_answer(answer)
    return []


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Answer questions against the indexed CV corpus."""
    original_query = message.content.strip()
    if not original_query:
        await cl.Message(content=EMPTY_QUERY_MESSAGE).send()
        return

    try:
        service = _get_query_service()
        candidate_refs = _get_candidate_refs()
        clarification = candidate_reference_clarification(
            original_query,
            candidate_refs,
        )
        if clarification is not None:
            await cl.Message(content=clarification).send()
            return

        query_text = rewrite_query_with_candidate_reference(
            original_query,
            candidate_refs,
        )
        conversation_context = _get_conversation_context()
        step: cl.Step | None = None

        msg = cl.Message(content="")
        await msg.send()

        async for node_name, state_update in service.async_stream(
            query_text,
            conversation_context=conversation_context or None,
        ):
            if step is None:
                step = cl.Step(
                    name=NODE_LABELS.get(node_name, node_name),
                    type="tool",
                )
                step.streaming = True
                await step.send()

            step.name = NODE_LABELS.get(node_name, node_name)
            step.output = _step_summary(node_name, state_update)

            if node_name == "finalize":
                step.streaming = False
                await step.update()
                await step.remove()
                final_text = state_update.get("final_text", "")
                if final_text:
                    full_cv = state_update.get("full_cv")
                    if full_cv is not None:
                        file_attachment = cl.File(
                            path=full_cv.pdf_path,
                            name=full_cv.source_file,
                            display="inline",
                        )
                        msg.content = final_text
                        msg.elements = [file_attachment]
                        await msg.update()
                    else:
                        for token in final_text.split(" "):
                            await msg.stream_token(token + " ")
                        await msg.update()
                cl.user_session.set(
                    "candidate_refs", _candidate_refs_from_state_update(state_update)
                )
                _set_conversation_context(
                    user_query=original_query,
                    state_update=state_update,
                )
            else:
                step.streaming = False
                await step.update()
    except (ValueError, TypeError, RuntimeError) as exc:
        await cl.Message(content=f"{RUNTIME_ERROR_MESSAGE}\n\nDetails: {exc}").send()
