"""Small formatting helpers for the Chainlit chat surface."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

from cv_screener.rag.nodes.finalize import format_answer
from cv_screener.rag.schema import AnswerCitation, AnswerOutput

if TYPE_CHECKING:
    from cv_screener.rag.service import RAGQueryResult

EMPTY_QUERY_MESSAGE = "Enter a recruiter-style question about the indexed CVs."
RUNTIME_ERROR_MESSAGE = "**Error:** I couldn't query the indexed CVs. Check that the PDFs are ingested and the local model services are running."


def format_chat_response(result: RAGQueryResult) -> str:
    """Render a RAG result for the Chainlit UI."""
    answer = result.state.get("answer")
    if isinstance(answer, AnswerOutput):
        return format_answer(answer)
    return result.final_text


def candidate_references_from_result(result: RAGQueryResult) -> list[str]:
    """Extract ordered candidate names from a grounded answer, if present."""
    answer = result.state.get("answer")
    if isinstance(answer, AnswerOutput):
        return candidate_references_from_answer(answer)
    full_cv = result.state.get("full_cv")
    candidate_name = getattr(full_cv, "candidate_name", None)
    if candidate_name:
        return [str(candidate_name)]
    targeted_lookup = result.state.get("targeted_lookup")
    candidate_names = getattr(targeted_lookup, "candidate_names", None)
    if isinstance(candidate_names, list) and candidate_names:
        return [str(name) for name in candidate_names if str(name).strip()]
    retrieved_chunks = result.state.get("retrieved_chunks", [])
    if isinstance(retrieved_chunks, list):
        ordered_candidates: list[str] = []
        seen: set[str] = set()
        for chunk in retrieved_chunks:
            candidate_name = getattr(chunk, "candidate_name", None)
            if not candidate_name or candidate_name in seen:
                continue
            seen.add(candidate_name)
            ordered_candidates.append(str(candidate_name))
        if ordered_candidates:
            return ordered_candidates
    return []


def candidate_references_from_answer(answer: AnswerOutput) -> list[str]:
    """Extract ordered candidate names from answer citations."""
    ordered_candidates: list[str] = []
    seen: set[str] = set()
    for citation in answer.citations:
        if not isinstance(citation, AnswerCitation):
            continue
        candidate_name = citation.candidate_name
        if not candidate_name or candidate_name in seen:
            continue
        seen.add(candidate_name)
        ordered_candidates.append(candidate_name)
    return ordered_candidates


def rewrite_query_with_candidate_reference(
    query_text: str,
    candidate_refs: list[str],
) -> str:
    """Rewrite simple ordinal references like 'second candidate' when possible."""
    if not candidate_refs:
        return query_text

    ordinal_match = re.search(
        r"\b(?:the\s+)?(?P<ordinal>first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?)\s+(?:candidate|one)(?:'s)?\b",
        query_text,
        re.IGNORECASE,
    )
    if ordinal_match is not None:
        index = _ordinal_to_index(ordinal_match.group("ordinal"))
        if index is not None and index < len(candidate_refs):
            candidate_name = candidate_refs[index]
            return (
                query_text[: ordinal_match.start()]
                + candidate_name
                + query_text[ordinal_match.end() :]
            )

    pronoun_match = re.search(
        r"\b(?:his|her|their|its)\s+(?P<kind>cv|resume|profile|document)\b",
        query_text,
        re.IGNORECASE,
    )
    if pronoun_match is not None and len(candidate_refs) == 1:
        candidate_name = candidate_refs[0]
        kind = pronoun_match.group("kind").casefold()
        if kind in {"cv", "resume", "document"}:
            return f"Give me the CV of {candidate_name}"
        if kind == "profile":
            return f"Summarize the profile of {candidate_name}"

    return query_text


def candidate_reference_clarification(
    query_text: str,
    candidate_refs: list[str],
) -> str | None:
    """Return a clarification prompt when a reference cannot be resolved cleanly."""
    if not query_text.strip():
        return None

    ordinal_match = re.search(
        r"\b(?:the\s+)?(?P<ordinal>first|second|third|fourth|fifth|\d+(?:st|nd|rd|th)?)\s+(?:candidate|one)(?:'s)?\b",
        query_text,
        re.IGNORECASE,
    )
    if ordinal_match is not None:
        index = _ordinal_to_index(ordinal_match.group("ordinal"))
        if index is None or index >= len(candidate_refs):
            return "Which candidate do you mean?"
        return None

    pronoun_match = re.search(
        r"\b(?:his|her|their|its)\s+(?:cv|resume|profile|document)\b",
        query_text,
        re.IGNORECASE,
    )
    if pronoun_match is not None and len(candidate_refs) != 1:
        return "Which candidate do you mean?"

    return None


def _ordinal_to_index(ordinal: str) -> int | None:
    normalized = ordinal.casefold()
    named_ordinals = {
        "first": 0,
        "second": 1,
        "third": 2,
        "fourth": 3,
        "fifth": 4,
    }
    if normalized in named_ordinals:
        return named_ordinals[normalized]
    match = re.fullmatch(r"(?P<value>\d+)(?:st|nd|rd|th)?", normalized)
    if match is None:
        return None
    return max(int(match.group("value")) - 1, 0)
