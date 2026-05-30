"""Parse CV PDFs into structured markdown using pymupdf4llm."""

from pathlib import Path
from typing import Any

import pymupdf4llm
from loguru import logger

from cv_screener.ingestion.schema import ParsedCV, ParsedPage


def _extract_page_chunks(pdf_path: Path) -> list[dict[str, Any]]:
    """Extract page chunks from a PDF, typed boundary for pymupdf4llm."""
    result = pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    if not isinstance(result, list):
        msg = f"pymupdf4llm returned {type(result).__name__}, expected list"
        raise TypeError(msg)
    return result


def parse_pdf(pdf_path: Path) -> ParsedCV:
    """Parse a single CV PDF into structured pages with markdown text."""
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)
    if pdf_path.suffix.lower() != ".pdf":
        msg = f"Expected a .pdf file, got {pdf_path.suffix}"
        raise ValueError(msg)

    logger.debug("Parsing PDF", path=str(pdf_path))
    page_chunks = _extract_page_chunks(pdf_path)

    metadata = page_chunks[0].get("metadata", {}) if page_chunks else {}
    title: str = metadata.get("title", "")

    pages = [
        ParsedPage(page_number=idx + 1, markdown=text)
        for idx, chunk in enumerate(page_chunks)
        if (text := chunk.get("text", "").strip())
    ]

    return ParsedCV(
        source_path=pdf_path,
        filename=pdf_path.stem,
        title=title,
        pages=pages,
    )


def parse_directory(directory: Path) -> list[ParsedCV]:
    """Parse all PDF files in a directory."""
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in directory", path=str(directory))
        return []

    logger.info("Parsing PDF directory", path=str(directory), count=len(pdf_files))
    results: list[ParsedCV] = []
    for pdf_path in pdf_files:
        logger.debug("Parsing PDF", path=str(pdf_path))
        results.append(parse_pdf(pdf_path))
    logger.info("Parsed PDFs", total=len(pdf_files), successful=len(results))
    return results
