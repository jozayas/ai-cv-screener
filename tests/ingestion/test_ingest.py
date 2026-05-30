from pathlib import Path

from cv_screener.ingestion.chunking.schema import Chunk
from cv_screener.ingestion.ingest import CVIngestionService, IngestionSummary
from cv_screener.ingestion.parsing.schema import ParsedCV, ParsedPage


def _make_parsed_cv(source_path: Path) -> ParsedCV:
    return ParsedCV(
        source_path=source_path,
        filename=source_path.stem,
        title="Candidate CV",
        pages=[
            ParsedPage(page_number=1, markdown="## **SUMMARY**\n\nBackend engineer.")
        ],
    )


def _make_chunk(source_file: str, chunk_index: int) -> Chunk:
    return Chunk.model_validate(
        {
            "candidate_name": "Jane Doe",
            "source_file": source_file,
            "document_title": "Candidate CV",
            "page": 1,
            "chunk_index": chunk_index,
            "section": "SUMMARY",
            "detected_skills": ["Python"],
            "detected_companies": [],
            "detected_universities": [],
            "email_addresses": [],
            "phone_numbers": [],
            "linkedin_urls": [],
            "github_urls": [],
            "text": f"Chunk {chunk_index}",
        }
    )


def test_ingest_parses_chunks_and_indexes_pdfs(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "pdfs"
    pdf_dir.mkdir()
    parsed_cvs = [
        _make_parsed_cv(pdf_dir / "candidate-1.pdf"),
        _make_parsed_cv(pdf_dir / "candidate-2.pdf"),
    ]
    chunks = [
        _make_chunk("candidate-1.pdf", 0),
        _make_chunk("candidate-2.pdf", 1),
        _make_chunk("candidate-2.pdf", 2),
    ]
    calls: list[tuple[str, object]] = []

    class FakeIndexer:
        def index_chunks(self, chunks: list[Chunk], *, reset: bool = False) -> None:
            calls.append(("index", (chunks, reset)))

    service = CVIngestionService(
        pdf_dir=pdf_dir,
        parser=lambda directory: calls.append(("parse", directory)) or parsed_cvs,
        chunker=lambda cvs: calls.append(("chunk", cvs)) or chunks,
        indexer=FakeIndexer(),
    )

    summary = service.ingest(reset=True)

    assert summary == IngestionSummary(pdf_count=2, chunk_count=3, reset=True)
    assert calls == [
        ("parse", pdf_dir),
        ("chunk", parsed_cvs),
        ("index", (chunks, True)),
    ]


def test_ingest_skips_indexing_when_no_pdfs_are_found(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "empty"
    pdf_dir.mkdir()
    calls: list[tuple[str, object]] = []

    class FakeIndexer:
        def index_chunks(self, chunks: list[Chunk], *, reset: bool = False) -> None:
            calls.append(("index", (chunks, reset)))

    service = CVIngestionService(
        pdf_dir=pdf_dir,
        parser=lambda directory: calls.append(("parse", directory)) or [],
        chunker=lambda cvs: calls.append(("chunk", cvs)) or [],
        indexer=FakeIndexer(),
    )

    summary = service.ingest(reset=False)

    assert summary == IngestionSummary(pdf_count=0, chunk_count=0, reset=False)
    assert calls == [("parse", pdf_dir)]
