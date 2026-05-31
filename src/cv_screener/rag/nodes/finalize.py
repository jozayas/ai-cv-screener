"""Final response formatting for the RAG graph."""

from __future__ import annotations

from typing import TYPE_CHECKING

from cv_screener.rag.schema import (
    AnswerOutput,
    BriefAnswerOutput,
    RouteDecision,
    RouteTarget,
)

if TYPE_CHECKING:
    from cv_screener.rag.state import RAGState


def finalize_node(state: RAGState) -> dict[str, str]:
    """Produce the user-visible response for terminal graph routes."""
    route = state.get("route")
    if not isinstance(route, RouteDecision):
        msg = "finalize state must include a route decision"
        raise TypeError(msg)

    if route.route is not RouteTarget.CV_QUERY:
        brief_answer = state.get("brief_answer")
        if not isinstance(brief_answer, BriefAnswerOutput):
            return {}
        return {"final_text": brief_answer.text}

    answer = state.get("answer")
    if not isinstance(answer, AnswerOutput):
        return {}
    return {"final_text": format_answer(answer)}


def format_answer(answer: AnswerOutput) -> str:
    """Render a grounded answer plus deduplicated source citations."""
    if answer.abstained:
        return answer.answer

    seen: set[tuple[str, int, str]] = set()
    citation_lines: list[str] = []
    for citation in answer.citations:
        citation_key = (citation.source_file, citation.page, citation.section)
        if citation_key in seen:
            continue
        seen.add(citation_key)
        citation_lines.append(
            f"- {citation.source_file} (page {citation.page}, {citation.section})"
        )

    return "\n\n".join([answer.answer, "Sources:\n" + "\n".join(citation_lines)])
