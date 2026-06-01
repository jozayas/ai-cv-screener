from pathlib import Path
from uuid import uuid4

from sqlalchemy import select

from cv_screener.cv_generation.content.schema import CVProfile
from cv_screener.cv_generation.content.yaml_io import write_cv_profile
from cv_screener.ingestion.chunking.schema import Chunk
from cv_screener.ingestion.indexing.points import build_payload, point_id_for_chunk
from cv_screener.ingestion.parsing.schema import ParsedCV, ParsedPage
from cv_screener.persistence.lookup import SQLiteLookupService
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
            "full_name": "Yaml Only",
            "email": "yaml@example.com",
            "phone": "+34123456789",
            "location": "Barcelona",
            "professional_summary": "Backend engineer.",
            "skills": ["Rust"],
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
                    "institution": "Yaml University",
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
            "detected_skills": ["Python", "FastAPI"],
            "detected_companies": [],
            "detected_universities": ["UPC"],
            "email_addresses": [],
            "phone_numbers": [],
            "linkedin_urls": [],
            "github_urls": [],
            "text": "Education\nUPC\nBackend engineer with Python experience.",
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
    assert stored_candidates[0].full_name == "Jane Doe"
    assert len(stored_documents) == 1
    assert stored_documents[0].source_file == pdf_path.name
    assert stored_documents[0].parsed_markdown == parsed.full_text
    assert stored_documents[0].yaml_path is None
    assert len(stored_chunks) == 1
    assert stored_chunks[0].chunk_id == point_id_for_chunk(chunk)
    assert stored_chunks[0].section_type == "SUMMARY"
    assert len(stored_skills) == 2
    assert len(stored_education) == 1
    assert len(stored_experience) == 0
    assert {row.name for row in stored_skills} == {"Python", "FastAPI"}
    assert stored_education[0].institution == "UPC"


def test_qdrant_payload_preserves_canonical_chunk_section() -> None:
    chunk = Chunk.model_validate(
        {
            "candidate_name": "Jane Doe",
            "source_file": "jane-doe.pdf",
            "document_title": "Jane Doe CV",
            "page": 1,
            "chunk_index": 0,
            "section": "EXPERIENCE",
            "detected_skills": [],
            "detected_companies": [],
            "detected_universities": [],
            "email_addresses": [],
            "phone_numbers": [],
            "linkedin_urls": [],
            "github_urls": [],
            "text": "Built APIs.",
        }
    )

    payload = build_payload(chunk)

    assert payload["section"] == "EXPERIENCE"


def test_sqlite_lookup_service_resolves_targeted_entities(tmp_path: Path) -> None:
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
            "section": "EDUCATION",
            "detected_skills": ["Python"],
            "detected_companies": [],
            "detected_universities": ["UPC"],
            "email_addresses": [],
            "phone_numbers": [],
            "linkedin_urls": [],
            "github_urls": [],
            "text": "Backend engineer with Python experience. Education: UPC.",
        }
    )

    sqlite_path = tmp_path / "canonical.db"
    repository = SQLiteCanonicalRepository(
        sqlite_path=sqlite_path, content_dir=content_dir
    )
    repository.persist(parsed_cvs=[parsed], chunks_to_store=[chunk])

    lookup = SQLiteLookupService(sqlite_path=sqlite_path)

    candidate = lookup.find_candidate_by_name("Jane Doe")
    skill_matches = lookup.find_candidates_by_skill("Python")
    education_matches = lookup.find_candidates_by_education("UPC")
    document = lookup.find_document_by_candidate_name("Jane Doe")
    chunks_for_candidate = lookup.fetch_chunks(
        candidate_ids=[str(candidate_id)],
        sections=["EDUCATION"],
    )

    assert candidate is not None
    assert candidate.full_name == "Jane Doe"
    assert [match.full_name for match in skill_matches] == ["Jane Doe"]
    assert [match.full_name for match in education_matches] == ["Jane Doe"]
    assert document is not None
    assert document.source_file == pdf_path.name
    assert document.parsed_markdown == parsed.full_text
    assert len(chunks_for_candidate) == 1
    assert chunks_for_candidate[0].section == "EDUCATION"


def test_sqlite_lookup_service_matches_partial_accented_names(
    tmp_path: Path,
) -> None:
    content_dir = tmp_path / "cvs_contents"
    pdf_dir = tmp_path / "cv_pdfs"
    content_dir.mkdir()
    pdf_dir.mkdir()

    candidate_id = uuid4()
    profile = CVProfile.model_validate(
        {
            "candidate_id": str(candidate_id),
            "full_name": "José Luis Zayas Alcaide",
            "email": "jose@example.com",
            "phone": "+34123456789",
            "location": "Málaga",
            "professional_summary": "AI engineer.",
            "skills": ["Python"],
            "experience": [
                {
                    "company": "Telefónica",
                    "role": "AI Engineer",
                    "start_date": "2024-01-01",
                    "end_date": None,
                    "summary": "Built RAG systems.",
                    "highlights": [],
                    "technologies": [],
                }
            ],
            "education": [
                {
                    "institution": "Universidad de Málaga",
                    "degree": "MSc",
                    "field_of_study": "Computer Science",
                    "graduation_year": 2023,
                }
            ],
        }
    )
    yaml_path = content_dir / f"jose-luis-{candidate_id}.yaml"
    write_cv_profile(yaml_path, profile)

    pdf_path = pdf_dir / f"jose-luis-{candidate_id}.pdf"
    parsed = ParsedCV(
        source_path=pdf_path,
        filename=pdf_path.stem,
        title="José Luis Zayas Alcaide CV",
        pages=[
            ParsedPage(
                page_number=1,
                markdown="## **SUMMARY**\n\nAI engineer with RAG experience.",
            )
        ],
    )
    chunk = Chunk.model_validate(
        {
            "candidate_name": "José Luis Zayas Alcaide",
            "source_file": pdf_path.name,
            "document_title": "José Luis Zayas Alcaide CV",
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
            "text": "AI engineer with RAG experience.",
        }
    )

    sqlite_path = tmp_path / "canonical.db"
    repository = SQLiteCanonicalRepository(
        sqlite_path=sqlite_path, content_dir=content_dir
    )
    repository.persist(parsed_cvs=[parsed], chunks_to_store=[chunk])

    lookup = SQLiteLookupService(sqlite_path=sqlite_path)

    match = lookup.find_candidate_by_name("Jose Luis")
    document = lookup.find_document_by_candidate_name("Jose Luis")

    assert match is not None
    assert match.full_name == "José Luis Zayas Alcaide"
    assert document is not None
    assert document.source_file == pdf_path.name
    assert document.parsed_markdown == parsed.full_text


def test_sqlite_lookup_service_uses_pdf_education_institution_variants(
    tmp_path: Path,
) -> None:
    content_dir = tmp_path / "cvs_contents"
    pdf_dir = tmp_path / "cv_pdfs"
    content_dir.mkdir()
    pdf_dir.mkdir()

    candidate_id = uuid4()
    profile = CVProfile.model_validate(
        {
            "candidate_id": str(candidate_id),
            "full_name": "Yaml Name",
            "email": "yaml@example.com",
            "phone": "+34123456789",
            "location": "Madrid",
            "professional_summary": "Yaml-only profile.",
            "skills": ["Rust"],
            "experience": [
                {
                    "company": "Yaml Company",
                    "role": "Yaml Role",
                    "start_date": "2020-01-01",
                    "end_date": None,
                    "summary": "Yaml-only experience.",
                    "highlights": [],
                    "technologies": [],
                }
            ],
            "education": [
                {
                    "institution": "Yaml University",
                    "degree": "MSc",
                    "field_of_study": "Computer Science",
                    "graduation_year": 2023,
                }
            ],
        }
    )
    yaml_path = content_dir / f"jose-luis-{candidate_id}.yaml"
    write_cv_profile(yaml_path, profile)

    pdf_path = pdf_dir / f"jose-luis-{candidate_id}.pdf"
    parsed = ParsedCV(
        source_path=pdf_path,
        filename=pdf_path.stem,
        title="José Luis Zayas Alcaide CV",
        pages=[
            ParsedPage(
                page_number=2,
                markdown=(
                    "## **Education**\n\n"
                    "**Universidad de Málaga (UMA)**, MS in Big Data & AI"
                ),
            )
        ],
    )
    chunk = Chunk.model_validate(
        {
            "candidate_name": "José Luis Zayas Alcaide",
            "source_file": pdf_path.name,
            "document_title": "José Luis Zayas Alcaide CV",
            "page": 2,
            "chunk_index": 0,
            "section": "header",
            "detected_skills": ["Python"],
            "detected_companies": [],
            "detected_universities": ["MS", "Universidad de Málaga"],
            "email_addresses": [],
            "phone_numbers": [],
            "linkedin_urls": [],
            "github_urls": [],
            "text": "Education\n**Universidad de Málaga (UMA)**, MS in Big Data & AI",
        }
    )

    sqlite_path = tmp_path / "canonical.db"
    repository = SQLiteCanonicalRepository(
        sqlite_path=sqlite_path, content_dir=content_dir
    )
    repository.persist(parsed_cvs=[parsed], chunks_to_store=[chunk])

    lookup = SQLiteLookupService(sqlite_path=sqlite_path)

    full_name_matches = lookup.find_candidates_by_education("Universidad de Malaga")
    acronym_matches = lookup.find_candidates_by_education("UMA")
    yaml_matches = lookup.find_candidates_by_education("Yaml University")
    degree_matches = lookup.find_candidates_by_education("MS")

    assert [match.full_name for match in full_name_matches] == [
        "José Luis Zayas Alcaide"
    ]
    assert [match.full_name for match in acronym_matches] == ["José Luis Zayas Alcaide"]
    assert yaml_matches == []
    assert degree_matches == []


def test_sqlite_lookup_service_matches_fuzzy_misspellings(
    tmp_path: Path,
) -> None:
    content_dir = tmp_path / "cvs_contents"
    pdf_dir = tmp_path / "cv_pdfs"
    content_dir.mkdir()
    pdf_dir.mkdir()

    candidate_id = uuid4()
    profile = CVProfile.model_validate(
        {
            "candidate_id": str(candidate_id),
            "full_name": "Alexandre Moreau",
            "email": "alexandre@example.com",
            "phone": "+33123456789",
            "location": "Paris",
            "professional_summary": "Platform engineer.",
            "skills": ["Python"],
            "experience": [
                {
                    "company": "Acme",
                    "role": "Platform Engineer",
                    "start_date": "2023-01-01",
                    "end_date": None,
                    "summary": "Built internal tooling.",
                    "highlights": [],
                    "technologies": [],
                }
            ],
            "education": [
                {
                    "institution": "Université de Paris",
                    "degree": "MSc",
                    "field_of_study": "Computer Science",
                    "graduation_year": 2022,
                }
            ],
        }
    )
    yaml_path = content_dir / f"alexandre-moreau-{candidate_id}.yaml"
    write_cv_profile(yaml_path, profile)

    pdf_path = pdf_dir / f"alexandre-moreau-{candidate_id}.pdf"
    parsed = ParsedCV(
        source_path=pdf_path,
        filename=pdf_path.stem,
        title="Alexandre Moreau CV",
        pages=[
            ParsedPage(
                page_number=1,
                markdown="## **SUMMARY**\n\nPlatform engineer.",
            )
        ],
    )
    chunk = Chunk.model_validate(
        {
            "candidate_name": "Alexandre Moreau",
            "source_file": pdf_path.name,
            "document_title": "Alexandre Moreau CV",
            "page": 1,
            "chunk_index": 0,
            "section": "SUMMARY",
            "detected_skills": [],
            "detected_companies": [],
            "detected_universities": [],
            "email_addresses": [],
            "phone_numbers": [],
            "linkedin_urls": [],
            "github_urls": [],
            "text": "Platform engineer.",
        }
    )
    repository = SQLiteCanonicalRepository(
        sqlite_path=tmp_path / "canonical.db",
        content_dir=content_dir,
    )
    repository.persist(parsed_cvs=[parsed], chunks_to_store=[chunk])

    lookup = SQLiteLookupService(sqlite_path=tmp_path / "canonical.db")

    match = lookup.find_candidate_by_name("Alexandre Morea")

    assert match is not None
    assert match.full_name == "Alexandre Moreau"
