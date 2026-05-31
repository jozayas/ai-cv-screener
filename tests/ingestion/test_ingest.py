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

    class FakeCanonicalStore:
        def reset_all(self) -> None:
            calls.append(("store_reset", None))

        def persist(
            self, *, parsed_cvs: list[ParsedCV], chunks_to_store: list[Chunk]
        ) -> None:
            calls.append(("store_persist", (parsed_cvs, chunks_to_store)))

    def parse(
        directory: Path, *, progress_callback: object | None = None
    ) -> list[ParsedCV]:
        _ = progress_callback
        calls.append(("parse", directory))
        return parsed_cvs

    def chunk(cvs: list[ParsedCV]) -> list[Chunk]:
        calls.append(("chunk", cvs))
        return chunks

    service = CVIngestionService(
        pdf_dir=pdf_dir,
        parser=parse,
        chunker=chunk,
        indexer=FakeIndexer(),
        canonical_store=FakeCanonicalStore(),
    )

    summary = service.ingest(reset=True)

    assert summary == IngestionSummary(pdf_count=2, chunk_count=3, reset=True)
    assert calls == [
        ("parse", pdf_dir),
        ("chunk", parsed_cvs),
        ("store_reset", None),
        ("store_persist", (parsed_cvs, chunks)),
        ("index", (chunks, True)),
    ]


def test_ingest_skips_indexing_when_no_pdfs_are_found(tmp_path: Path) -> None:
    pdf_dir = tmp_path / "empty"
    pdf_dir.mkdir()
    calls: list[tuple[str, object]] = []

    class FakeIndexer:
        def index_chunks(self, chunks: list[Chunk], *, reset: bool = False) -> None:
            calls.append(("index", (chunks, reset)))

    def parse(
        directory: Path, *, progress_callback: object | None = None
    ) -> list[ParsedCV]:
        _ = progress_callback
        calls.append(("parse", directory))
        return []

    def chunk(cvs: list[ParsedCV]) -> list[Chunk]:
        calls.append(("chunk", cvs))
        return []

    service = CVIngestionService(
        pdf_dir=pdf_dir,
        parser=parse,
        chunker=chunk,
        indexer=FakeIndexer(),
    )

    summary = service.ingest(reset=False)

    assert summary == IngestionSummary(pdf_count=0, chunk_count=0, reset=False)
    assert calls == [("parse", pdf_dir)]
