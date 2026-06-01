"""SQLite-backed targeted lookup helpers for deterministic CV queries."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from sqlalchemy import create_engine, select

from cv_screener.persistence.schema import (
    candidates,
    chunks,
    documents,
    education,
    skills,
)
from cv_screener.retrieval.schema import RetrievedChunk

_DEFAULT_MIN_NAME_MATCH_SCORE = 0.72
_PARENTHETICAL_PATTERN = re.compile(r"\(([^()]*)\)")


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    cleaned = re.sub(r"[^a-z0-9]+", " ", stripped.casefold())
    return " ".join(cleaned.split())


def _institution_lookup_keys(value: str) -> set[str]:
    normalized = _normalize(value)
    if not normalized:
        return set()
    keys = {normalized}
    without_parentheticals = _PARENTHETICAL_PATTERN.sub(" ", value)
    base_key = _normalize(without_parentheticals)
    if base_key:
        keys.add(base_key)
    keys.update(
        parenthetical
        for match in _PARENTHETICAL_PATTERN.finditer(value)
        if (parenthetical := _normalize(match.group(1)))
    )
    return keys


def _name_match_score(
    query_normalized: str,
    query_tokens: list[str],
    candidate_normalized: str,
    candidate_tokens: list[str],
) -> float:
    if candidate_normalized == query_normalized:
        return 1.0
    if query_normalized and (
        query_normalized in candidate_normalized
        or candidate_normalized in query_normalized
    ):
        return 0.96
    if query_tokens and all(token in candidate_tokens for token in query_tokens):
        return 0.93
    if query_tokens and candidate_tokens:
        overlap = len(set(query_tokens) & set(candidate_tokens))
        if overlap:
            token_ratio = overlap / max(len(query_tokens), len(candidate_tokens))
            sequence_ratio = SequenceMatcher(
                None, query_normalized, candidate_normalized
            ).ratio()
            return max(token_ratio, sequence_ratio)
    return SequenceMatcher(None, query_normalized, candidate_normalized).ratio()


@dataclass(frozen=True)
class CandidateMatch:
    """Matched candidate from the canonical SQLite store."""

    candidate_id: str
    full_name: str


@dataclass(frozen=True)
class DocumentMatch:
    """Full document metadata for a matched candidate."""

    candidate_id: str
    full_name: str
    source_file: str
    document_title: str
    pdf_path: str
    parsed_markdown: str


class SQLiteLookupService:
    """Deterministic lookup service over canonical SQLite data."""

    def __init__(
        self,
        *,
        sqlite_path: Path,
        candidate_name_min_score: float = _DEFAULT_MIN_NAME_MATCH_SCORE,
    ) -> None:
        """Create a lookup service backed by the canonical SQLite database."""
        self._engine = create_engine(f"sqlite:///{Path(sqlite_path)}", future=True)
        self._candidate_name_min_score = candidate_name_min_score

    def find_candidate_by_name(self, name: str) -> CandidateMatch | None:
        """Resolve a candidate by exact, partial, or fuzzy normalized name."""
        matches = self.find_candidates_by_name(name)
        if len(matches) == 1:
            return matches[0]
        return None

    def find_candidates_by_name(self, name: str) -> list[CandidateMatch]:
        """Return candidates whose normalized names best match the query."""
        normalized = _normalize(name)
        tokens = normalized.split()
        if not tokens:
            return []

        stmt = select(candidates.c.candidate_id, candidates.c.full_name)
        with self._engine.begin() as conn:
            rows = conn.execute(stmt).all()

        scored_matches: list[tuple[float, CandidateMatch]] = []
        for row in rows:
            candidate_normalized = _normalize(row.full_name)
            candidate_tokens = candidate_normalized.split()
            score = _name_match_score(
                normalized,
                tokens,
                candidate_normalized,
                candidate_tokens,
            )
            if score < self._candidate_name_min_score:
                continue
            scored_matches.append(
                (
                    score,
                    CandidateMatch(
                        candidate_id=row.candidate_id,
                        full_name=row.full_name,
                    ),
                )
            )

        if not scored_matches:
            return []

        scored_matches.sort(key=lambda item: item[0], reverse=True)
        best_score = scored_matches[0][0]
        return [match for score, match in scored_matches if score == best_score]

    def find_candidates_by_skill(self, skill_name: str) -> list[CandidateMatch]:
        """Return all candidates that list the given skill."""
        normalized = _normalize(skill_name)
        stmt = (
            select(candidates.c.candidate_id, candidates.c.full_name)
            .select_from(
                skills.join(
                    candidates,
                    skills.c.candidate_id == candidates.c.candidate_id,
                )
            )
            .where(skills.c.normalized_name == normalized)
        )
        with self._engine.begin() as conn:
            rows = conn.execute(stmt).all()
        seen: set[str] = set()
        matches: list[CandidateMatch] = []
        for row in rows:
            if row.candidate_id in seen:
                continue
            seen.add(row.candidate_id)
            matches.append(
                CandidateMatch(candidate_id=row.candidate_id, full_name=row.full_name)
            )
        return matches

    def find_candidates_by_education(self, institution: str) -> list[CandidateMatch]:
        """Return all candidates who studied at the given institution."""
        lookup_keys = _institution_lookup_keys(institution)
        if not lookup_keys:
            return []
        stmt = select(
            candidates.c.candidate_id,
            candidates.c.full_name,
            education.c.institution,
            education.c.normalized_institution,
        ).select_from(
            education.join(
                candidates,
                education.c.candidate_id == candidates.c.candidate_id,
            )
        )
        with self._engine.begin() as conn:
            rows = conn.execute(stmt).all()
        seen: set[str] = set()
        matches: list[CandidateMatch] = []
        for row in rows:
            row_keys = _institution_lookup_keys(row.institution)
            row_keys.add(_normalize(row.normalized_institution))
            if lookup_keys.isdisjoint(row_keys):
                continue
            if row.candidate_id in seen:
                continue
            seen.add(row.candidate_id)
            matches.append(
                CandidateMatch(candidate_id=row.candidate_id, full_name=row.full_name)
            )
        return matches

    def find_document_by_candidate_name(self, name: str) -> DocumentMatch | None:
        """Return the full document metadata for a candidate, or ``None``."""
        candidate = self.find_candidate_by_name(name)
        if candidate is None:
            return None
        stmt = (
            select(
                documents.c.candidate_id,
                candidates.c.full_name,
                documents.c.source_file,
                documents.c.document_title,
                documents.c.pdf_path,
                documents.c.parsed_markdown,
            )
            .select_from(
                documents.join(
                    candidates,
                    documents.c.candidate_id == candidates.c.candidate_id,
                )
            )
            .where(documents.c.candidate_id == candidate.candidate_id)
        )
        with self._engine.begin() as conn:
            row = conn.execute(stmt).first()
        if row is None:
            return None
        return DocumentMatch(
            candidate_id=row.candidate_id,
            full_name=row.full_name,
            source_file=row.source_file,
            document_title=row.document_title,
            pdf_path=row.pdf_path,
            parsed_markdown=row.parsed_markdown or "",
        )

    def fetch_chunks(
        self,
        *,
        candidate_ids: list[str],
        sections: list[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return chunks for the given candidates, optionally filtered by section."""
        if not candidate_ids:
            return []
        stmt = (
            select(
                chunks.c.candidate_name,
                documents.c.source_file,
                documents.c.document_title,
                chunks.c.page_start,
                chunks.c.section_type,
                chunks.c.text,
            )
            .select_from(
                chunks.join(
                    documents,
                    chunks.c.document_id == documents.c.document_id,
                )
            )
            .where(chunks.c.candidate_id.in_(candidate_ids))
            .order_by(
                documents.c.source_file, chunks.c.page_start, chunks.c.chunk_index
            )
        )
        if sections:
            stmt = stmt.where(chunks.c.section_type.in_(sections))
        with self._engine.begin() as conn:
            rows = conn.execute(stmt).all()
        return [
            RetrievedChunk(
                candidate_name=row.candidate_name,
                source_file=row.source_file,
                document_title=row.document_title,
                page=row.page_start,
                section=row.section_type,
                text=row.text,
                score=1.0,
                rank=index,
            )
            for index, row in enumerate(rows, start=1)
        ]
