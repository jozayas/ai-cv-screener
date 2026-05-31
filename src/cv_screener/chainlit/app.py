"""Chainlit chat UI for the CV screener."""

from __future__ import annotations

from typing import Any, cast

import chainlit as cl
from chainlit.config import config

from cv_screener.chainlit.ui import EMPTY_QUERY_MESSAGE, RUNTIME_ERROR_MESSAGE
from cv_screener.cli.dependencies import (
    RAGQueryServiceProtocol,
    build_rag_query_service,
)

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
        service = build_rag_query_service()
        cl.user_session.set("rag_query_service", service)
    return service


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Answer questions against the indexed CV corpus."""
    query_text = message.content.strip()
    if not query_text:
        await cl.Message(content=EMPTY_QUERY_MESSAGE).send()
        return

    try:
        service = _get_query_service()
        step: cl.Step | None = None

        msg = cl.Message(content="")
        await msg.send()

        async for node_name, state_update in service.async_stream(query_text):
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
                    for token in final_text.split(" "):
                        await msg.stream_token(token + " ")
                    await msg.update()
            else:
                step.streaming = False
                await step.update()
    except (ValueError, TypeError, RuntimeError) as exc:
        await cl.Message(content=f"{RUNTIME_ERROR_MESSAGE}\n\nDetails: {exc}").send()


