"""SQLite-backed repository for canonical extracted CV data."""

from __future__ import annotations

import pathlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import Connection, Engine, create_engine, delete, insert, select

from cv_screener.cv_generation.content.yaml_io import load_cv_profile
from cv_screener.ingestion.indexing.points import (
    document_id_for_chunk,
    point_id_for_chunk,
)
from cv_screener.persistence.schema import (
    candidates,
    chunks,
    documents,
    education,
    experience,
    metadata,
    skills,
)

if TYPE_CHECKING:
    from pathlib import Path

    from cv_screener.cv_generation.content.schema import CVProfile
    from cv_screener.ingestion.chunking.schema import Chunk
    from cv_screener.ingestion.parsing.schema import ParsedCV

_CANDIDATE_ID_PATTERN = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12})$"
)


class CanonicalStoreProtocol(Protocol):
    """Behavior needed by ingestion for canonical persistence."""

    def reset_all(self) -> None:
        """Reset canonical store rows before re-ingestion."""

    def persist(
        self, *, parsed_cvs: list[ParsedCV], chunks_to_store: list[Chunk]
    ) -> None:
        """Persist parsed CVs and chunks as canonical extracted data."""


@dataclass(frozen=True)
class _MatchedProfile:
    profile: CVProfile
    yaml_path: Path


def _normalize(value: str) -> str:
    return " ".join(value.casefold().split())


def _candidate_id_from_filename(filename_stem: str) -> str | None:
    match = _CANDIDATE_ID_PATTERN.search(filename_stem)
    if match is None:
        return None
    return str(UUID(match.group(1)))


def _fallback_candidate_id(cv: ParsedCV) -> str:
    return str(uuid5(NAMESPACE_URL, f"candidate\n{cv.source_path.name}"))


class SQLiteCanonicalRepository:
    """Canonical SQL store that persists candidate records and CV chunks."""

    def __init__(self, *, sqlite_path: Path, content_dir: Path) -> None:
        """Bind database and YAML source paths for canonical persistence."""
        sqlite_path = pathlib.Path(sqlite_path)
        content_dir = pathlib.Path(content_dir)
        self._sqlite_path = sqlite_path
        self._content_dir = content_dir
        self._engine = create_engine(f"sqlite:///{sqlite_path}", future=True)

    @property
    def engine(self) -> Engine:
        """Expose engine for tests and diagnostics."""
        return self._engine

    def ensure_schema(self) -> None:
        """Create canonical tables and indexes if missing."""
        self._sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        metadata.create_all(self._engine)

    def reset_all(self) -> None:
        """Clear canonical tables while preserving schema."""
        self.ensure_schema()
        with self._engine.begin() as conn:
            _ = conn.execute(delete(chunks))
            _ = conn.execute(delete(documents))
            _ = conn.execute(delete(skills))
            _ = conn.execute(delete(education))
            _ = conn.execute(delete(experience))
            _ = conn.execute(delete(candidates))

    def persist(
        self, *, parsed_cvs: list[ParsedCV], chunks_to_store: list[Chunk]
    ) -> None:
        """Persist parsed CVs and chunk rows in one transaction."""
        self.ensure_schema()
        profiles_by_candidate_id = self._load_profiles_by_candidate_id()
        chunks_by_source_file = self._group_chunks_by_source_file(chunks_to_store)

        with self._engine.begin() as conn:
            for cv in parsed_cvs:
                matched = self._match_profile(cv, profiles_by_candidate_id)
                candidate_id = (
                    str(matched.profile.candidate_id)
                    if matched is not None
                    else _candidate_id_from_filename(cv.filename)
                    or _fallback_candidate_id(cv)
                )
                source_file = cv.source_path.name
                source_chunks = chunks_by_source_file.get(source_file, [])
                document_id = (
                    document_id_for_chunk(source_chunks[0])
                    if source_chunks
                    else str(uuid5(NAMESPACE_URL, f"{source_file}\n{cv.title.strip()}"))
                )

                self._upsert_candidate(
                    conn=conn,
                    candidate_id=candidate_id,
                    matched=matched,
                    fallback_name=source_chunks[0].candidate_name
                    if source_chunks
                    else None,
                )
                self._replace_structured_rows(
                    conn=conn,
                    candidate_id=candidate_id,
                    profile=matched.profile if matched is not None else None,
                )
                self._upsert_document(
                    conn=conn,
                    document_id=document_id,
                    candidate_id=candidate_id,
                    cv=cv,
                    yaml_path=matched.yaml_path if matched is not None else None,
                )
                self._replace_document_chunks(
                    conn=conn,
                    candidate_id=candidate_id,
                    document_id=document_id,
                    source_chunks=source_chunks,
                )

    def _load_profiles_by_candidate_id(self) -> dict[str, _MatchedProfile]:
        profiles: dict[str, _MatchedProfile] = {}
        if not self._content_dir.exists():
            return profiles
        for yaml_path in sorted(self._content_dir.glob("*.yaml")):
            profile = load_cv_profile(yaml_path)
            profiles[str(profile.candidate_id)] = _MatchedProfile(
                profile=profile,
                yaml_path=yaml_path,
            )
        return profiles

    @staticmethod
    def _group_chunks_by_source_file(
        chunks_to_store: list[Chunk],
    ) -> dict[str, list[Chunk]]:
        grouped: dict[str, list[Chunk]] = {}
        for chunk in chunks_to_store:
            grouped.setdefault(chunk.source_file, []).append(chunk)
        return grouped

    @staticmethod
    def _match_profile(
        cv: ParsedCV,
        profiles_by_candidate_id: dict[str, _MatchedProfile],
    ) -> _MatchedProfile | None:
        candidate_id = _candidate_id_from_filename(cv.filename)
        if candidate_id is None:
            return None
        return profiles_by_candidate_id.get(candidate_id)

    def _upsert_candidate(
        self,
        *,
        conn: Connection,
        candidate_id: str,
        matched: _MatchedProfile | None,
        fallback_name: str | None,
    ) -> None:
        profile = matched.profile if matched is not None else None
        values = {
            "candidate_id": candidate_id,
            "full_name": profile.full_name
            if profile is not None
            else (fallback_name or "Unknown"),
            "email": profile.email if profile is not None else None,
            "phone": profile.phone if profile is not None else None,
            "location": profile.location if profile is not None else None,
            "professional_summary": (
                profile.professional_summary if profile is not None else None
            ),
        }
        _ = conn.execute(
            delete(candidates).where(candidates.c.candidate_id == candidate_id)
        )
        _ = conn.execute(insert(candidates).values(values))

    def _replace_structured_rows(
        self,
        *,
        conn: Connection,
        candidate_id: str,
        profile: CVProfile | None,
    ) -> None:
        _ = conn.execute(delete(skills).where(skills.c.candidate_id == candidate_id))
        _ = conn.execute(
            delete(education).where(education.c.candidate_id == candidate_id)
        )
        _ = conn.execute(
            delete(experience).where(experience.c.candidate_id == candidate_id)
        )
        if profile is None:
            return

        if profile.skills:
            _ = conn.execute(
                insert(skills),
                [
                    {
                        "candidate_id": candidate_id,
                        "sort_order": index,
                        "name": skill_name,
                        "normalized_name": _normalize(skill_name),
                    }
                    for index, skill_name in enumerate(profile.skills)
                ],
            )
        if profile.education:
            _ = conn.execute(
                insert(education),
                [
                    {
                        "candidate_id": candidate_id,
                        "sort_order": index,
                        "institution": entry.institution,
                        "normalized_institution": _normalize(entry.institution),
                        "degree": entry.degree,
                        "field_of_study": entry.field_of_study,
                        "graduation_year": entry.graduation_year,
                    }
                    for index, entry in enumerate(profile.education)
                ],
            )
        if profile.experience:
            _ = conn.execute(
                insert(experience),
                [
                    {
                        "candidate_id": candidate_id,
                        "sort_order": index,
                        "company": entry.company,
                        "normalized_company": _normalize(entry.company),
                        "role": entry.role,
                        "normalized_role": _normalize(entry.role),
                        "start_date": entry.start_date.isoformat(),
                        "end_date": entry.end_date.isoformat()
                        if entry.end_date
                        else None,
                        "summary": entry.summary,
                    }
                    for index, entry in enumerate(profile.experience)
                ],
            )

    def _upsert_document(
        self,
        *,
        conn: Connection,
        document_id: str,
        candidate_id: str,
        cv: ParsedCV,
        yaml_path: Path | None,
    ) -> None:
        _ = conn.execute(
            delete(documents).where(documents.c.document_id == document_id)
        )
        _ = conn.execute(
            insert(documents).values(
                document_id=document_id,
                candidate_id=candidate_id,
                source_file=cv.source_path.name,
                source_path=str(cv.source_path),
                document_title=cv.title.strip(),
                pdf_path=str(cv.source_path),
                yaml_path=str(yaml_path) if yaml_path is not None else None,
            )
        )

    def _replace_document_chunks(
        self,
        *,
        conn: Connection,
        candidate_id: str,
        document_id: str,
        source_chunks: list[Chunk],
    ) -> None:
        existing_chunk_ids = conn.execute(
            select(chunks.c.chunk_id).where(chunks.c.document_id == document_id)
        ).scalars()
        existing_chunk_ids_list = list(existing_chunk_ids)
        if existing_chunk_ids_list:
            _ = conn.execute(delete(chunks).where(chunks.c.document_id == document_id))
        if not source_chunks:
            return

        _ = conn.execute(
            insert(chunks),
            [
                {
                    "chunk_id": point_id_for_chunk(chunk),
                    "document_id": document_id,
                    "candidate_id": candidate_id,
                    "chunk_index": chunk.chunk_index,
                    "page_start": chunk.page,
                    "page_end": chunk.page,
                    "section_type": chunk.section,
                    "candidate_name": chunk.candidate_name,
                    "text": chunk.text,
                }
                for chunk in source_chunks
            ],
        )
