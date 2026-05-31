"""Parse CV PDFs into structured markdown using pymupdf4llm."""

import os
import warnings
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pymupdf4llm
from loguru import logger

from cv_screener.ingestion.parsing.schema import ParsedCV, ParsedPage

type ProgressCallback = Callable[[int, int], None]


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


def parse_directory(
    directory: Path,
    *,
    progress_callback: ProgressCallback | None = None,
) -> list[ParsedCV]:
    """Parse all PDF files in a directory."""
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    warnings.filterwarnings(
        "ignore",
        message="The `resume_download` argument is deprecated and ignored.*",
        category=UserWarning,
    )
    directory = Path(directory)
    if not directory.is_dir():
        raise NotADirectoryError(directory)

    pdf_files = sorted(directory.glob("*.pdf"))
    if not pdf_files:
        logger.warning("No PDF files found in directory", path=str(directory))
        return []

    logger.info("Parsing PDF directory", path=str(directory), count=len(pdf_files))
    workers = min(8, len(pdf_files))
    parsed_by_index: list[ParsedCV | None] = [None] * len(pdf_files)
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_to_index = {
            executor.submit(parse_pdf, pdf_path): idx
            for idx, pdf_path in enumerate(pdf_files)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            parsed_by_index[idx] = future.result()
            completed += 1
            if progress_callback is not None:
                progress_callback(completed, len(pdf_files))
    results = [parsed for parsed in parsed_by_index if parsed is not None]
    logger.info("Parsed PDFs", total=len(pdf_files), successful=len(results))
    return results
