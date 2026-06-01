"""Tests for BM25 sparse vector encoding."""

from cv_screener.ingestion.indexing.bm25 import BM25Encoder, BM25EncoderProtocol


class FakeBM25Encoder:
    """Fake encoder implementing the protocol for unit tests."""

    def __init__(self, model_name: str = "Qdrant/bm25") -> None:
        del model_name

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        return [{1: 1.0, 2: 0.5} for _ in texts]


def test_bm25_encoder_satisfies_protocol() -> None:
    assert issubclass(BM25Encoder, BM25EncoderProtocol)


def test_fake_encoder_satisfies_protocol() -> None:
    assert isinstance(FakeBM25Encoder(), BM25EncoderProtocol)


def test_fake_encoder_returns_aligned_results() -> None:
    encoder = FakeBM25Encoder()
    result = encoder.encode(["hello world", "foo bar"])
    assert len(result) == 2
    assert result[0] == {1: 1.0, 2: 0.5}
    assert result[1] == {1: 1.0, 2: 0.5}


def test_fake_encoder_returns_empty_for_empty_input() -> None:
    encoder = FakeBM25Encoder()
    assert encoder.encode([]) == []
