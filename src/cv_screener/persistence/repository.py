"""SQLite-backed repository for canonical extracted CV data."""

from __future__ import annotations

import pathlib
import re
import unicodedata
from collections import defaultdict
from typing import TYPE_CHECKING, Protocol
from uuid import NAMESPACE_URL, UUID, uuid5

from sqlalchemy import Connection, Engine, create_engine, delete, insert, select, update

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

    from cv_screener.ingestion.chunking.schema import Chunk
    from cv_screener.ingestion.parsing.schema import ParsedCV

_CANDIDATE_ID_PATTERN = re.compile(
    r"([0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}-"
    r"[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12})$"
)
_UNIVERSITY_PATTERN = re.compile(
    r"\b((?:Universidad|Universitat|University|Université|Ecole|École|Institute|"
    r"Instituto)[^,\n\r\u2013-]*(?:\([A-Za-z0-9 .&+-]{2,16}\))?)",
    re.IGNORECASE,
)
_MARKDOWN_TOKEN_PATTERN = re.compile(r"[*_`]+")


class CanonicalStoreProtocol(Protocol):
    """Behavior needed by ingestion for canonical persistence."""

    def reset_all(self) -> None:
        """Reset canonical store rows before re-ingestion."""

    def persist(
        self, *, parsed_cvs: list[ParsedCV], chunks_to_store: list[Chunk]
    ) -> None:
        """Persist parsed CVs and chunks as canonical extracted data."""


def _normalize(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value)
    stripped = "".join(char for char in decomposed if not unicodedata.combining(char))
    cleaned = re.sub(r"[^a-z0-9]+", " ", stripped.casefold())
    return " ".join(cleaned.split())


def _clean_extracted_value(value: str) -> str:
    cleaned = _MARKDOWN_TOKEN_PATTERN.sub("", value)
    return " ".join(cleaned.strip(" \t\r\n,.;:[]{}").split())


def _dedupe_extracted_values(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        cleaned = _clean_extracted_value(value)
        if not cleaned:
            continue
        normalized = _normalize(cleaned)
        if normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(cleaned)
    return deduped


def _looks_like_institution(value: str) -> bool:
    cleaned = _clean_extracted_value(value)
    normalized = _normalize(cleaned)
    if re.search(
        r"\b(universidad|universitat|university|universite|ecole|institute|instituto)\b",
        normalized,
    ):
        return True
    return bool(re.fullmatch(r"[A-Z0-9]{3,8}", cleaned))


def _skill_names_from_chunks(chunks_to_extract: list[Chunk]) -> list[str]:
    return _dedupe_extracted_values(
        [skill for chunk in chunks_to_extract for skill in chunk.detected_skills]
    )


def _education_institutions_from_chunks(chunks_to_extract: list[Chunk]) -> list[str]:
    detected = [
        university
        for chunk in chunks_to_extract
        if _is_education_context(chunk)
        for university in chunk.detected_universities
        if _looks_like_institution(university)
    ]
    parsed_from_education_text = [
        match.group(1)
        for chunk in chunks_to_extract
        if _is_education_context(chunk)
        for match in _UNIVERSITY_PATTERN.finditer(chunk.text)
    ]
    return _dedupe_extracted_values([*detected, *parsed_from_education_text])


def _is_education_context(chunk: Chunk) -> bool:
    if chunk.section.casefold() == "education":
        return True
    cleaned_text = _MARKDOWN_TOKEN_PATTERN.sub("", chunk.text).lstrip()
    return cleaned_text.casefold().startswith("education")


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
        """Bind database and legacy content path for canonical persistence."""
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
        self._ensure_document_markdown_column()
        self._backfill_document_markdown()

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
        chunks_by_source_file = self._group_chunks_by_source_file(chunks_to_store)

        with self._engine.begin() as conn:
            for cv in parsed_cvs:
                candidate_id = _candidate_id_from_filename(
                    cv.filename
                ) or _fallback_candidate_id(cv)
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
                    fallback_name=source_chunks[0].candidate_name
                    if source_chunks
                    else cv.title.strip(),
                )
                self._replace_structured_rows(
                    conn=conn,
                    candidate_id=candidate_id,
                    source_chunks=source_chunks,
                )
                self._upsert_document(
                    conn=conn,
                    document_id=document_id,
                    candidate_id=candidate_id,
                    cv=cv,
                )
                self._replace_document_chunks(
                    conn=conn,
                    candidate_id=candidate_id,
                    document_id=document_id,
                    source_chunks=source_chunks,
                )

    @staticmethod
    def _group_chunks_by_source_file(
        chunks_to_store: list[Chunk],
    ) -> dict[str, list[Chunk]]:
        grouped: dict[str, list[Chunk]] = {}
        for chunk in chunks_to_store:
            grouped.setdefault(chunk.source_file, []).append(chunk)
        return grouped

    def _ensure_document_markdown_column(self) -> None:
        """Add the parsed markdown column to legacy databases if needed."""
        with self._engine.begin() as conn:
            columns = {
                row[1]
                for row in conn.exec_driver_sql("PRAGMA table_info(documents)").all()
            }
            if "parsed_markdown" not in columns:
                _ = conn.exec_driver_sql(
                    "ALTER TABLE documents ADD COLUMN parsed_markdown TEXT"
                )

    def _backfill_document_markdown(self) -> None:
        """Populate missing parsed markdown from stored chunks when possible."""
        with self._engine.begin() as conn:
            document_rows = conn.execute(
                select(documents.c.document_id, documents.c.parsed_markdown)
            ).all()
            missing_document_ids = [
                row.document_id
                for row in document_rows
                if not isinstance(row.parsed_markdown, str) or not row.parsed_markdown
            ]
            if not missing_document_ids:
                return

            chunk_rows = conn.execute(
                select(
                    chunks.c.document_id,
                    chunks.c.page_start,
                    chunks.c.chunk_index,
                    chunks.c.text,
                )
                .where(chunks.c.document_id.in_(missing_document_ids))
                .order_by(
                    chunks.c.document_id, chunks.c.page_start, chunks.c.chunk_index
                )
            ).all()
            grouped_chunks: dict[str, list[str]] = defaultdict(list)
            for row in chunk_rows:
                grouped_chunks[row.document_id].append(row.text)

            for document_id in missing_document_ids:
                markdown = "\n\n".join(grouped_chunks.get(document_id, []))
                if not markdown:
                    continue
                _ = conn.execute(
                    update(documents)
                    .where(documents.c.document_id == document_id)
                    .values(parsed_markdown=markdown)
                )

    def _upsert_candidate(
        self,
        *,
        conn: Connection,
        candidate_id: str,
        fallback_name: str | None,
    ) -> None:
        values = {
            "candidate_id": candidate_id,
            "full_name": fallback_name or "Unknown",
            "email": None,
            "phone": None,
            "location": None,
            "professional_summary": None,
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
        source_chunks: list[Chunk],
    ) -> None:
        _ = conn.execute(delete(skills).where(skills.c.candidate_id == candidate_id))
        _ = conn.execute(
            delete(education).where(education.c.candidate_id == candidate_id)
        )
        _ = conn.execute(
            delete(experience).where(experience.c.candidate_id == candidate_id)
        )
        skill_names = _skill_names_from_chunks(source_chunks)
        if skill_names:
            _ = conn.execute(
                insert(skills),
                [
                    {
                        "candidate_id": candidate_id,
                        "sort_order": index,
                        "name": skill_name,
                        "normalized_name": _normalize(skill_name),
                    }
                    for index, skill_name in enumerate(skill_names)
                ],
            )
        institutions = _education_institutions_from_chunks(source_chunks)
        if institutions:
            _ = conn.execute(
                insert(education),
                [
                    {
                        "candidate_id": candidate_id,
                        "sort_order": index,
                        "institution": institution,
                        "normalized_institution": _normalize(institution),
                        "degree": "",
                        "field_of_study": "",
                        "graduation_year": 0,
                    }
                    for index, institution in enumerate(institutions)
                ],
            )

    def _upsert_document(
        self,
        *,
        conn: Connection,
        document_id: str,
        candidate_id: str,
        cv: ParsedCV,
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
                parsed_markdown=cv.full_text,
                yaml_path=None,
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
