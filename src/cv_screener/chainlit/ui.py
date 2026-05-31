"""Small formatting helpers for the Chainlit chat surface."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cv_screener.rag.nodes.finalize import format_answer
from cv_screener.rag.schema import AnswerOutput

if TYPE_CHECKING:
    from cv_screener.rag.service import RAGQueryResult

WELCOME_MESSAGE = (
    "Ask a recruiter-style question about the indexed CVs.\n\n"
    "Examples:\n"
    "- Who has Python backend experience?\n"
    "- Which candidates mention Kubernetes?\n"
    "- Summarize the profile of Ada Lovelace."
)
EMPTY_QUERY_MESSAGE = "Enter a recruiter-style question about the indexed CVs."
RUNTIME_ERROR_MESSAGE = "**Error:** I couldn't query the indexed CVs. Check that the PDFs are ingested and the local model services are running."


def format_chat_response(result: RAGQueryResult) -> str:
    """Render a RAG result for the Chainlit UI."""
    answer = result.state.get("answer")
    if isinstance(answer, AnswerOutput):
        return format_answer(answer)
    return result.final_text
