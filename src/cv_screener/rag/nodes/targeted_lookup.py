"""Deterministic SQLite-first lookup nodes for targeted CV queries."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Protocol

from cv_screener.rag.schema import FullCVOutput, TargetedLookupOutput
from cv_screener.retrieval.schema import RetrievedChunk

if TYPE_CHECKING:
    from cv_screener.persistence.lookup import CandidateMatch, DocumentMatch
    from cv_screener.rag.state import RAGState


class LookupServiceProtocol(Protocol):
    """Structural interface for deterministic SQLite-first CV lookups."""

    def find_candidate_by_name(self, name: str) -> CandidateMatch | None:
        """Find a candidate by exact full-name match."""
        ...

    def find_candidates_by_name(self, name: str) -> list[CandidateMatch]:
        """Find all candidates matching the provided name."""
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
        sections=None if lookup.response_mode == "profile" else lookup.sections or None,
    )
    if lookup.response_mode == "list_candidates" and lookup.evidence_term:
        chunks = _matching_chunks_by_candidate(
            chunks,
            candidate_names=lookup.candidate_names,
            evidence_term=lookup.evidence_term,
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


def return_profile_cv_node(
    state: RAGState,
    *,
    lookup_service: LookupServiceProtocol,
) -> dict[str, FullCVOutput]:
    """Resolve a targeted single-candidate profile query to the matched CV document."""
    lookup = state.get("targeted_lookup")
    if not isinstance(lookup, TargetedLookupOutput):
        msg = "profile CV state must include targeted_lookup output"
        raise TypeError(msg)
    if lookup.response_mode != "profile" or len(lookup.candidate_names) != 1:
        msg = "profile CV state must include a single targeted profile candidate"
        raise ValueError(msg)

    candidate_name = lookup.candidate_names[0]
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


def full_cv_to_chunks_node(state: RAGState) -> dict[str, list[RetrievedChunk]]:
    """Convert a resolved full CV document into a single answer-context chunk."""
    full_cv = state.get("full_cv")
    if not isinstance(full_cv, FullCVOutput):
        msg = "profile context state must include full_cv output"
        raise TypeError(msg)

    return {
        "retrieved_chunks": [
            RetrievedChunk(
                candidate_name=full_cv.candidate_name,
                source_file=full_cv.source_file,
                document_title=full_cv.document_title,
                page=1,
                section="FULL_CV",
                text=full_cv.parsed_markdown,
                score=1.0,
                rank=1,
            )
        ]
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
            fallback_to_semantic=False,
        )

    skill_match = re.search(
        r"\bwho (?:has experience with\s+(.+?)|(?:has|knows)\s+(.+?))[?.!]*$",
        user_query,
        re.IGNORECASE,
    )
    if skill_match is not None:
        raw_skill = next(group for group in skill_match.groups() if group is not None)
        skill = _normalize_skill_lookup(raw_skill)
        blocked_terms = ("background", "worked", "leadership")
        if not any(term in skill.casefold() for term in blocked_terms):
            candidates = lookup_service.find_candidates_by_skill(skill)
            return TargetedLookupOutput(
                candidate_ids=[candidate.candidate_id for candidate in candidates],
                candidate_names=[candidate.full_name for candidate in candidates],
                sections=["SKILLS", "EXPERIENCE", "PROJECTS"],
                evidence_term=skill,
                response_mode="list_candidates",
                fallback_to_semantic=False,
            )

    profile_match = re.search(
        r"\b(?:profile of|summarize(?: the profile of)?|summarize)\s+(.+?)[?.!]*$",
        user_query,
        re.IGNORECASE,
    )
    if profile_match is not None:
        candidate_name = profile_match.group(1).strip()
        candidates = lookup_service.find_candidates_by_name(candidate_name)
        if not candidates:
            return TargetedLookupOutput(
                response_mode="clarify",
                clarification_message=(
                    f"I couldn't find a candidate matching '{candidate_name}'. "
                    "Please provide the full name."
                ),
                fallback_to_semantic=False,
            )
        if len(candidates) > 1:
            names = ", ".join(candidate.full_name for candidate in candidates)
            return TargetedLookupOutput(
                candidate_ids=[candidate.candidate_id for candidate in candidates],
                candidate_names=[candidate.full_name for candidate in candidates],
                response_mode="clarify",
                clarification_message=(
                    f"I found multiple candidates matching '{candidate_name}': "
                    f"{names}. Which one do you want?"
                ),
                fallback_to_semantic=False,
            )
        candidate = candidates[0]
        return TargetedLookupOutput(
            candidate_ids=[candidate.candidate_id],
            candidate_names=[candidate.full_name],
            sections=[],
            response_mode="profile",
            fallback_to_semantic=False,
        )

    return TargetedLookupOutput(response_mode="profile", fallback_to_semantic=True)


def _normalize_skill_lookup(raw_skill: str) -> str:
    skill = raw_skill.strip()
    experience_suffix = re.search(
        r"^(?P<skill>.+?)\s+experience$",
        skill,
        re.IGNORECASE,
    )
    if experience_suffix is not None:
        return experience_suffix.group("skill").strip()
    return skill


def _matching_chunks_by_candidate(
    chunks: list[RetrievedChunk],
    *,
    candidate_names: list[str],
    evidence_term: str,
) -> list[RetrievedChunk]:
    """Keep one chunk per candidate whose text contains the targeted evidence term."""
    normalized_term = _normalize_evidence_text(evidence_term)
    if not normalized_term:
        return chunks

    ordered_candidate_names = [
        name for name in candidate_names if _normalize_evidence_text(name)
    ]
    candidate_order = {
        _normalize_evidence_text(name): index
        for index, name in enumerate(ordered_candidate_names)
    }
    matches: dict[str, RetrievedChunk] = {}
    for chunk in chunks:
        candidate_name = chunk.candidate_name
        if not candidate_name:
            continue
        normalized_candidate_name = _normalize_evidence_text(candidate_name)
        if normalized_candidate_name not in candidate_order:
            continue
        if normalized_term not in _normalize_evidence_text(chunk.text):
            continue
        matches.setdefault(normalized_candidate_name, chunk)

    return [
        matches[name]
        for name, _index in sorted(candidate_order.items(), key=lambda item: item[1])
        if name in matches
    ]


def _normalize_evidence_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.casefold()).strip()


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
