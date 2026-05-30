"""Payload mapping for indexed chunks."""

from cv_screener.ingestion.chunking.schema import Chunk
from cv_screener.ingestion.indexing.ids import document_id_for_chunk


def build_payload(chunk: Chunk) -> dict[str, str | int | list[str] | None]:
    """Build a Qdrant payload from a chunk."""
    return {
        "document_id": document_id_for_chunk(chunk),
        "candidate_name": chunk.candidate_name,
        "source_file": chunk.source_file,
        "document_title": chunk.document_title,
        "page": chunk.page,
        "chunk_index": chunk.chunk_index,
        "section": chunk.section,
        "detected_skills": chunk.detected_skills,
        "detected_companies": chunk.detected_companies,
        "detected_universities": chunk.detected_universities,
        "email_addresses": chunk.email_addresses,
        "phone_numbers": chunk.phone_numbers,
        "linkedin_urls": chunk.linkedin_urls,
        "github_urls": chunk.github_urls,
        "text": chunk.text,
    }
