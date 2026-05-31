"""Final response formatting for the RAG graph."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from cv_screener.rag.schema import (
    AnswerCitation,
    AnswerOutput,
    BriefAnswerOutput,
    FullCVOutput,
    RouteDecision,
    RouteTarget,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from cv_screener.rag.state import RAGState


def finalize_node(state: RAGState) -> dict[str, object]:
    """Produce the user-visible response for terminal graph routes."""
    route = state.get("route")
    if not isinstance(route, RouteDecision):
        msg = "finalize state must include a route decision"
        raise TypeError(msg)

    result: dict[str, object] = {
        "nodes_executed": [*state.get("nodes_executed", []), "finalize"]
    }

    if route.route is RouteTarget.FULL_CV:
        return _finalize_full_cv(state, result)

    if (
        route.route is not RouteTarget.CV_QUERY
        and route.route is not RouteTarget.TARGETED_LOOKUP
    ):
        brief_answer = state.get("brief_answer")
        if isinstance(brief_answer, BriefAnswerOutput):
            result["final_text"] = brief_answer.text
        return result

    targeted_result = _finalize_targeted_lookup(state, result, route.route)
    if targeted_result is not None:
        return targeted_result

    answer = state.get("answer")
    if isinstance(answer, AnswerOutput):
        result["final_text"] = format_answer(answer)
        result["answer"] = answer
    targeted_lookup = state.get("targeted_lookup")
    if targeted_lookup is not None:
        result["targeted_lookup"] = targeted_lookup
    return result


def _finalize_full_cv(
    state: RAGState,
    result: dict[str, object],
) -> dict[str, object]:
    full_cv = state.get("full_cv")
    if isinstance(full_cv, FullCVOutput):
        markdown = full_cv.parsed_markdown.strip()
        text_parts = [
            f"{full_cv.candidate_name} - {full_cv.source_file}",
            markdown,
            f"PDF: {full_cv.pdf_path}",
        ]
        result["final_text"] = "\n\n".join(part for part in text_parts if part)
        result["full_cv"] = full_cv
        result["targeted_lookup"] = state.get("targeted_lookup")
    return result


def _finalize_targeted_lookup(
    state: RAGState,
    result: dict[str, object],
    route: RouteTarget,
) -> dict[str, object] | None:
    if route is not RouteTarget.TARGETED_LOOKUP:
        return None

    targeted_lookup = state.get("targeted_lookup")
    response_mode = getattr(targeted_lookup, "response_mode", None)
    if response_mode == "clarify":
        clarification_message = getattr(targeted_lookup, "clarification_message", None)
        if isinstance(clarification_message, str):
            result["final_text"] = clarification_message
            result["targeted_lookup"] = targeted_lookup
        return result

    if response_mode == "list_candidates":
        retrieved_chunks = state.get("retrieved_chunks", [])
        result["final_text"] = _format_candidate_list_response(
            retrieved_chunks,
            targeted_lookup,
        )
        result["targeted_lookup"] = targeted_lookup
        return result

    return None


def _citation_label(citation: AnswerCitation) -> str:
    """Human-readable label for a citation."""
    name = citation.candidate_name or citation.source_file.removesuffix(".pdf")
    return f"{name} - {citation.source_file} (page {citation.page}, {citation.section})"


def format_answer(answer: AnswerOutput) -> str:
    """Render a grounded answer with deduplicated source citations ranked by relevance."""
    if answer.abstained:
        return answer.answer

    seen: set[tuple[str, int, str]] = set()
    citation_lines: list[str] = []
    for citation in answer.citations:
        citation_key = (citation.source_file, citation.page, citation.section)
        if citation_key not in seen:
            seen.add(citation_key)
            citation_lines.append(f"[{citation.rank}] {_citation_label(citation)}")

    return "\n\n".join([answer.answer, "Sources:\n" + "\n".join(citation_lines)])


def _format_candidate_list_response(
    retrieved_chunks: Sequence[Any],
    targeted_lookup: object,
) -> str:
    candidate_names = getattr(targeted_lookup, "candidate_names", None)
    if isinstance(candidate_names, list) and candidate_names:
        names = [str(name) for name in candidate_names if str(name).strip()]
    else:
        names = []
        seen_names: set[str] = set()
        for chunk in retrieved_chunks:
            candidate_name = getattr(chunk, "candidate_name", None)
            if not candidate_name or candidate_name in seen_names:
                continue
            seen_names.add(candidate_name)
            names.append(str(candidate_name))

    if not names:
        return "I don't have enough information in the indexed CVs to answer that."

    citations: list[str] = []
    seen_sources: set[tuple[str, int, str]] = set()
    for chunk in retrieved_chunks:
        source_file = getattr(chunk, "source_file", "")
        page = getattr(chunk, "page", 0)
        section = getattr(chunk, "section", "")
        candidate_name = getattr(chunk, "candidate_name", None)
        key = (source_file, page, section)
        if key in seen_sources or not source_file or page <= 0 or not section:
            continue
        if candidate_name and candidate_name in names:
            seen_sources.add(key)
            citations.append(
                f"[{len(citations) + 1}] {candidate_name} - {source_file} (page {page}, {section})"
            )

    answer_text = "Candidates who match the query: " + ", ".join(
        f"{name} [{index}]" for index, name in enumerate(names, start=1)
    )
    if citations:
        return "\n\n".join([answer_text, "Sources:\n" + "\n".join(citations)])
    return answer_text
