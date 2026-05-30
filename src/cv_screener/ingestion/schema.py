"""Lightweight shared models for ingestion."""

from dataclasses import dataclass


@dataclass(frozen=True)
class IngestionSummary:
    """Stable summary returned by the ingest orchestration."""

    pdf_count: int
    chunk_count: int
    reset: bool
