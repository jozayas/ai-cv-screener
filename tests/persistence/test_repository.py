from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from cv_screener.cv_generation.content.schema import CVProfile
from cv_screener.cv_generation.content.yaml_io import write_cv_profile
from cv_screener.ingestion.chunking.schema import Chunk
from cv_screener.ingestion.indexing.points import point_id_for_chunk
from cv_screener.ingestion.parsing.schema import ParsedCV, ParsedPage
from cv_screener.persistence.repository import SQLiteCanonicalRepository
from cv_screener.persistence.schema import (
    candidates,
    chunks,
    documents,
    education,
    experience,
    skills,
)


def test_repository_persists_candidate_document_and_chunks(tmp_path: Path) -> None:
    content_dir = tmp_path / "cvs_contents"
    pdf_dir = tmp_path / "cv_pdfs"
    content_dir.mkdir()
    pdf_dir.mkdir()

    candidate_id = uuid4()
    profile = CVProfile.model_validate(
        {
            "candidate_id": str(candidate_id),
            "full_name": "Jane Doe",
            "email": "jane@example.com",
            "phone": "+34123456789",
            "location": "Barcelona",
            "professional_summary": "Backend engineer.",
            "skills": ["Python", "FastAPI"],
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
                    "graduation_year": 2020,
                }
            ],
        }
    )
    yaml_path = content_dir / f"jane-doe-{candidate_id}.yaml"
    write_cv_profile(yaml_path, profile)

    pdf_path = pdf_dir / f"jane-doe-{candidate_id}.pdf"
    parsed = ParsedCV(
        source_path=pdf_path,
        filename=pdf_path.stem,
        title="Jane Doe CV",
        pages=[
            ParsedPage(page_number=1, markdown="## **SUMMARY**\n\nBackend engineer.")
        ],
    )
    chunk = Chunk.model_validate(
        {
            "candidate_name": "Jane Doe",
            "source_file": pdf_path.name,
            "document_title": "Jane Doe CV",
            "page": 1,
            "chunk_index": 0,
            "section": "SUMMARY",
            "detected_skills": ["Python"],
            "detected_companies": [],
            "detected_universities": [],
            "email_addresses": [],
            "phone_numbers": [],
            "linkedin_urls": [],
            "github_urls": [],
            "text": "Backend engineer with Python experience.",
        }
    )

    repository = SQLiteCanonicalRepository(
        sqlite_path=tmp_path / "canonical.db",
        content_dir=content_dir,
    )
    repository.persist(parsed_cvs=[parsed], chunks_to_store=[chunk])

    with repository.engine.begin() as conn:
        stored_candidates = conn.execute(select(candidates)).all()
        stored_documents = conn.execute(select(documents)).all()
        stored_chunks = conn.execute(select(chunks)).all()
        stored_skills = conn.execute(select(skills)).all()
        stored_education = conn.execute(select(education)).all()
        stored_experience = conn.execute(select(experience)).all()

    assert len(stored_candidates) == 1
    assert stored_candidates[0].candidate_id == str(candidate_id)
    assert len(stored_documents) == 1
    assert stored_documents[0].source_file == pdf_path.name
    assert len(stored_chunks) == 1
    assert stored_chunks[0].chunk_id == point_id_for_chunk(chunk)
    assert stored_chunks[0].section_type == "SUMMARY"
    assert len(stored_skills) == 2
    assert len(stored_education) == 1
    assert len(stored_experience) == 1
