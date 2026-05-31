"""SQLite-backed targeted lookup helpers for deterministic CV queries."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
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


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    cleaned = re.sub(r"[^a-z0-9]+", " ", stripped.casefold())
    return " ".join(cleaned.split())


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

    def __init__(self, *, sqlite_path: Path) -> None:
        """Create a lookup service backed by the canonical SQLite database."""
        self._engine = create_engine(f"sqlite:///{Path(sqlite_path)}", future=True)

    def find_candidate_by_name(self, name: str) -> CandidateMatch | None:
        """Resolve a candidate by exact or partial normalized name."""
        matches = self.find_candidates_by_name(name)
        if len(matches) == 1:
            return matches[0]
        return None

    def find_candidates_by_name(self, name: str) -> list[CandidateMatch]:
        """Return candidates whose normalized names match the query tokens."""
        normalized = _normalize(name)
        tokens = normalized.split()
        if not tokens:
            return []

        stmt = select(candidates.c.candidate_id, candidates.c.full_name)
        with self._engine.begin() as conn:
            rows = conn.execute(stmt).all()

        exact_matches: list[CandidateMatch] = []
        partial_matches: list[CandidateMatch] = []
        for row in rows:
            candidate_normalized = _normalize(row.full_name)
            candidate_tokens = candidate_normalized.split()
            if candidate_normalized == normalized:
                exact_matches.append(
                    CandidateMatch(
                        candidate_id=row.candidate_id,
                        full_name=row.full_name,
                    )
                )
                continue
            if all(token in candidate_tokens for token in tokens):
                partial_matches.append(
                    CandidateMatch(
                        candidate_id=row.candidate_id,
                        full_name=row.full_name,
                    )
                )

        if exact_matches:
            return exact_matches
        if len(partial_matches) == 1:
            return partial_matches
        return partial_matches

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
        return [
            CandidateMatch(candidate_id=row.candidate_id, full_name=row.full_name)
            for row in rows
        ]

    def find_candidates_by_education(self, institution: str) -> list[CandidateMatch]:
        """Return all candidates who studied at the given institution."""
        normalized = _normalize(institution)
        stmt = (
            select(candidates.c.candidate_id, candidates.c.full_name)
            .select_from(
                education.join(
                    candidates,
                    education.c.candidate_id == candidates.c.candidate_id,
                )
            )
            .where(education.c.normalized_institution == normalized)
        )
        with self._engine.begin() as conn:
            rows = conn.execute(stmt).all()
        return [
            CandidateMatch(candidate_id=row.candidate_id, full_name=row.full_name)
            for row in rows
        ]

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
