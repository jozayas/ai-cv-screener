"""SQLAlchemy Core schema for canonical CV persistence."""

from __future__ import annotations

from sqlalchemy import Column, ForeignKey, Index, Integer, MetaData, Table, Text

metadata = MetaData()

candidates = Table(
    "candidates",
    metadata,
    Column("candidate_id", Text, primary_key=True),
    Column("full_name", Text, nullable=False),
    Column("email", Text, nullable=True),
    Column("phone", Text, nullable=True),
    Column("location", Text, nullable=True),
    Column("professional_summary", Text, nullable=True),
)

documents = Table(
    "documents",
    metadata,
    Column("document_id", Text, primary_key=True),
    Column(
        "candidate_id",
        Text,
        ForeignKey("candidates.candidate_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("source_file", Text, nullable=False, unique=True),
    Column("source_path", Text, nullable=False),
    Column("document_title", Text, nullable=False),
    Column("pdf_path", Text, nullable=False),
    Column("yaml_path", Text, nullable=True),
)

chunks = Table(
    "chunks",
    metadata,
    Column("chunk_id", Text, primary_key=True),
    Column(
        "document_id",
        Text,
        ForeignKey("documents.document_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column(
        "candidate_id",
        Text,
        ForeignKey("candidates.candidate_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("chunk_index", Integer, nullable=False),
    Column("page_start", Integer, nullable=False),
    Column("page_end", Integer, nullable=False),
    Column("section_type", Text, nullable=False),
    Column("candidate_name", Text, nullable=True),
    Column("text", Text, nullable=False),
)

education = Table(
    "education",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "candidate_id",
        Text,
        ForeignKey("candidates.candidate_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("sort_order", Integer, nullable=False),
    Column("institution", Text, nullable=False),
    Column("normalized_institution", Text, nullable=False),
    Column("degree", Text, nullable=False),
    Column("field_of_study", Text, nullable=False),
    Column("graduation_year", Integer, nullable=False),
)

experience = Table(
    "experience",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "candidate_id",
        Text,
        ForeignKey("candidates.candidate_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("sort_order", Integer, nullable=False),
    Column("company", Text, nullable=False),
    Column("normalized_company", Text, nullable=False),
    Column("role", Text, nullable=False),
    Column("normalized_role", Text, nullable=False),
    Column("start_date", Text, nullable=False),
    Column("end_date", Text, nullable=True),
    Column("summary", Text, nullable=False),
)

skills = Table(
    "skills",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "candidate_id",
        Text,
        ForeignKey("candidates.candidate_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("sort_order", Integer, nullable=False),
    Column("name", Text, nullable=False),
    Column("normalized_name", Text, nullable=False),
)

_ = Index("idx_skills_normalized_name", skills.c.normalized_name)
_ = Index("idx_education_normalized_institution", education.c.normalized_institution)
_ = Index("idx_experience_normalized_company", experience.c.normalized_company)
_ = Index("idx_experience_normalized_role", experience.c.normalized_role)
_ = Index("idx_chunks_candidate_id", chunks.c.candidate_id)
_ = Index("idx_chunks_document_id", chunks.c.document_id)
