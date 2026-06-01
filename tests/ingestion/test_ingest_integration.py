from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import pytest
from qdrant_client import QdrantClient
from sqlalchemy import select

from cv_screener.cv_generation.content.schema import CVProfile
from cv_screener.cv_generation.content.yaml_io import write_cv_profile
from cv_screener.ingestion.chunking.schema import Chunk
from cv_screener.ingestion.indexing.points import point_id_for_chunk
from cv_screener.ingestion.indexing.qdrant import QdrantChunkIndexer
from cv_screener.ingestion.indexing.schema import QdrantIndexConfig
from cv_screener.ingestion.ingest import CVIngestionService
from cv_screener.ingestion.parsing.schema import ParsedCV, ParsedPage
from cv_screener.persistence.repository import SQLiteCanonicalRepository
from cv_screener.persistence.schema import (
    candidates,
    documents,
)
from cv_screener.persistence.schema import (
    chunks as chunks_table,
)

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


class FakeEmbeddingModel:
    def embed(
        self,
        documents: str | Iterable[str],
        batch_size: int = 256,
        parallel: int | None = None,
    ) -> list[list[float]]:
        del batch_size, parallel
        texts = [documents] if isinstance(documents, str) else list(documents)
        return [[0.1, 0.2, 0.3] for _ in texts]


@pytest.mark.integration
def test_ingestion_persists_sqlite_and_indexes_qdrant(tmp_path: Path) -> None:
    candidate_id = uuid4()
    content_dir = tmp_path / "cvs_contents"
    pdf_dir = tmp_path / "cv_pdfs"
    content_dir.mkdir()
    pdf_dir.mkdir()

    profile = CVProfile.model_validate(
        {
            "candidate_id": str(candidate_id),
            "full_name": "Marta Alvarez",
            "email": "marta@example.com",
            "phone": "+34111222333",
            "location": "Barcelona",
            "professional_summary": "Backend engineer with Python experience.",
            "skills": ["Python"],
            "experience": [
                {
                    "company": "Leadtech",
                    "role": "Backend Engineer",
                    "start_date": "2022-01-01",
                    "end_date": None,
                    "summary": "Built APIs.",
                    "highlights": [],
                    "technologies": [],
                }
            ],
            "education": [
                {
                    "institution": "UPC",
                    "degree": "BSc",
                    "field_of_study": "Computer Science",
                    "graduation_year": 2021,
                }
            ],
        }
    )
    yaml_path = content_dir / f"marta-alvarez-{candidate_id}.yaml"
    write_cv_profile(yaml_path, profile)

    pdf_path = pdf_dir / f"marta-alvarez-{candidate_id}.pdf"
    parsed_cv = ParsedCV(
        source_path=pdf_path,
        filename=pdf_path.stem,
        title="Marta Alvarez CV",
        pages=[
            ParsedPage(
                page_number=1, markdown="## **SUMMARY**\n\nPython backend engineer."
            )
        ],
    )
    chunk = Chunk(
        candidate_name="Marta Alvarez",
        source_file=pdf_path.name,
        document_title="Marta Alvarez CV",
        page=1,
        chunk_index=0,
        section="SUMMARY",
        detected_skills=["Python"],
        detected_companies=[],
        detected_universities=[],
        email_addresses=["marta@example.com"],
        phone_numbers=[],
        linkedin_urls=[],
        github_urls=[],
        text="Python backend engineer.",
    )

    sqlite_repository = SQLiteCanonicalRepository(
        sqlite_path=tmp_path / "cv_screener.db",
        content_dir=content_dir,
    )
    collection_name = f"cv_chunks_ingest_integration_{uuid4().hex[:8]}"
    qdrant_client = QdrantClient(url="http://localhost:6333")
    qdrant_indexer = QdrantChunkIndexer(
        config=QdrantIndexConfig(collection_name=collection_name, vector_size=3),
        client=qdrant_client,
        embedding_model=FakeEmbeddingModel(),
    )

    def parse(
        directory: Path,
        *,
        progress_callback: object | None = None,
    ) -> list[ParsedCV]:
        _ = (directory, progress_callback)
        return [parsed_cv]

    def chunk_corpus(cvs: list[ParsedCV]) -> list[Chunk]:
        _ = cvs
        return [chunk]

    service = CVIngestionService(
        pdf_dir=pdf_dir,
        parser=parse,
        chunker=chunk_corpus,
        canonical_store=sqlite_repository,
        indexer=qdrant_indexer,
    )

    summary = service.ingest(reset=True)
    assert summary.pdf_count == 1
    assert summary.chunk_count == 1

    stored_points = qdrant_client.retrieve(
        collection_name=collection_name,
        ids=[UUID(point_id_for_chunk(chunk))],
        with_payload=True,
        with_vectors=False,
    )
    assert len(stored_points) == 1

    with sqlite_repository.engine.begin() as conn:
        candidate_rows = conn.execute(select(candidates.c.candidate_id)).all()
        document_rows = conn.execute(select(documents.c.document_id)).all()
        chunk_rows = conn.execute(select(chunks_table.c.chunk_id)).all()
    assert len(candidate_rows) == 1
    assert candidate_rows[0].candidate_id == str(candidate_id)
    assert len(document_rows) == 1
    assert len(chunk_rows) == 1
