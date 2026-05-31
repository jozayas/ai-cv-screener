"""Chainlit chat UI for the CV screener."""

from __future__ import annotations

from typing import cast

import chainlit as cl

from cv_screener.chainlit.ui import (
    EMPTY_QUERY_MESSAGE,
    WELCOME_MESSAGE,
    format_chat_response,
)
from cv_screener.cli.dependencies import (
    RAGQueryServiceProtocol,
    build_rag_query_service,
)


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


@cl.on_chat_start
async def on_chat_start() -> None:
    """Show a concise introduction when a chat starts."""
    _ = await cl.Message(content=WELCOME_MESSAGE).send()


@cl.on_message
async def on_message(message: cl.Message) -> None:
    """Answer questions against the indexed CV corpus."""
    query_text = message.content.strip()
    if not query_text:
        _ = await cl.Message(content=EMPTY_QUERY_MESSAGE).send()
        return

    service = _get_query_service()
    result = await cl.make_async(service.run)(query_text)
    _ = await cl.Message(content=format_chat_response(result)).send()
