"""Deterministic SQLite-first lookup nodes for targeted CV queries."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

from cv_screener.rag.schema import FullCVOutput, TargetedLookupOutput

if TYPE_CHECKING:
    from cv_screener.persistence.lookup import CandidateMatch, DocumentMatch
    from cv_screener.rag.state import RAGState
    from cv_screener.retrieval.schema import RetrievedChunk


class LookupServiceProtocol(Protocol):
    """Structural interface for deterministic SQLite-first CV lookups."""

    def find_candidate_by_name(self, name: str) -> CandidateMatch | None:
        """Find a candidate by exact full-name match."""
        ...

    def find_candidates_by_skill(self, skill_name: str) -> list[CandidateMatch]:
        """Find all candidates listing the given skill."""
        ...

    def find_candidates_by_education(self, institution: str) -> list[CandidateMatch]:
        """Find all candidates who studied at the given institution."""
        ...

    def find_document_by_candidate_name(self, name: str) -> DocumentMatch | None:
        """Return full document metadata for a candidate."""
        ...

    def fetch_chunks(
        self, *, candidate_ids: list[str], sections: list[str] | None = None
    ) -> list[RetrievedChunk]:
        """Fetch chunks for the given candidates, filtered by section."""
        ...


def targeted_lookup_node(
    state: RAGState,
    *,
    lookup_service: LookupServiceProtocol,
) -> dict[str, TargetedLookupOutput]:
    """Resolve a user query into targeted candidate IDs and sections."""
    user_query = state.get("user_query")
    if not isinstance(user_query, str) or not user_query.strip():
        msg = "targeted lookup state must include a non-empty user_query"
        raise ValueError(msg)

    lookup = _resolve_lookup(user_query, lookup_service=lookup_service)
    return {"targeted_lookup": lookup}


def hydrate_chunks_node(
    state: RAGState,
    *,
    lookup_service: LookupServiceProtocol,
) -> dict[str, list[RetrievedChunk]]:
    """Fetch full-text chunks for the targeted candidate IDs."""
    lookup = state.get("targeted_lookup")
    if not isinstance(lookup, TargetedLookupOutput):
        msg = "hydrate state must include targeted_lookup output"
        raise TypeError(msg)
    chunks = lookup_service.fetch_chunks(
        candidate_ids=lookup.candidate_ids,
        sections=lookup.sections or None,
    )
    return {"retrieved_chunks": chunks}


def return_cv_node(
    state: RAGState,
    *,
    lookup_service: LookupServiceProtocol,
) -> dict[str, FullCVOutput]:
    """Resolve a user query to a full CV document reference."""
    user_query = state.get("user_query")
    if not isinstance(user_query, str) or not user_query.strip():
        msg = "return_cv state must include a non-empty user_query"
        raise ValueError(msg)

    candidate_name = _extract_candidate_name_for_cv(user_query)
    match = lookup_service.find_document_by_candidate_name(candidate_name)
    if match is None:
        msg = f"No CV found for {candidate_name}."
        raise ValueError(msg)
    return {
        "full_cv": FullCVOutput(
            candidate_name=match.full_name,
            source_file=match.source_file,
            document_title=match.document_title,
            pdf_path=match.pdf_path,
            parsed_markdown=match.parsed_markdown,
        )
    }


def _resolve_lookup(
    user_query: str,
    *,
    lookup_service: LookupServiceProtocol,
) -> TargetedLookupOutput:
    education_match = re.search(
        r"\bgraduated from\s+(.+?)[?.!]*$", user_query, re.IGNORECASE
    )
    if education_match is not None:
        institution = education_match.group(1).strip()
        candidates = lookup_service.find_candidates_by_education(institution)
        return TargetedLookupOutput(
            candidate_ids=[candidate.candidate_id for candidate in candidates],
            candidate_names=[candidate.full_name for candidate in candidates],
            sections=["EDUCATION"],
            response_mode="list_candidates",
            fallback_to_semantic=not candidates,
        )

    skill_match = re.search(
        r"\bwho (?:has|knows)\s+(.+?)[?.!]*$", user_query, re.IGNORECASE
    )
    if skill_match is not None:
        skill = skill_match.group(1).strip()
        blocked_terms = ("experience", "background", "worked", "leadership")
        if not any(term in skill.casefold() for term in blocked_terms):
            candidates = lookup_service.find_candidates_by_skill(skill)
            return TargetedLookupOutput(
                candidate_ids=[candidate.candidate_id for candidate in candidates],
                candidate_names=[candidate.full_name for candidate in candidates],
                sections=["SKILLS", "EXPERIENCE", "PROJECTS"],
                response_mode="list_candidates",
                fallback_to_semantic=not candidates,
            )

    profile_match = re.search(
        r"\b(?:profile of|summarize(?: the profile of)?|summarize)\s+(.+?)[?.!]*$",
        user_query,
        re.IGNORECASE,
    )
    if profile_match is not None:
        candidate_name = profile_match.group(1).strip()
        candidate = lookup_service.find_candidate_by_name(candidate_name)
        return TargetedLookupOutput(
            candidate_ids=[] if candidate is None else [candidate.candidate_id],
            candidate_names=[] if candidate is None else [candidate.full_name],
            sections=["PROFILE", "SUMMARY", "EXPERIENCE", "EDUCATION", "SKILLS"],
            response_mode="profile",
            fallback_to_semantic=candidate is None,
        )

    return TargetedLookupOutput(response_mode="profile", fallback_to_semantic=True)


def _extract_candidate_name_for_cv(user_query: str) -> str:
    match = re.search(
        r"\b(?:cv|resume)\s+(?:of|for)\s+(.+?)[?.!]*$",
        user_query,
        re.IGNORECASE,
    )
    if match is not None:
        return match.group(1).strip()
    fallback = re.search(
        r"\bgive me (?:the )?(?:cv|resume)\s+of\s+(.+?)[?.!]*$",
        user_query,
        re.IGNORECASE,
    )
    if fallback is not None:
        return fallback.group(1).strip()
    return user_query.strip()
