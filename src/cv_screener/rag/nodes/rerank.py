"""Cross-encoder reranking for retrieved CV chunks."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Protocol, cast

from cv_screener.rag.schema import PlannerOutput

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cv_screener.rag.state import RAGState
    from cv_screener.retrieval.schema import RetrievedChunk


FAST_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
HIGH_QUALITY_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_RERANKER_MODEL = FAST_RERANKER_MODEL
FALLBACK_RERANKER_MODEL = FAST_RERANKER_MODEL


class CrossEncoderProtocol(Protocol):
    """Subset of the sentence-transformers cross-encoder API used here."""

    def predict(
        self,
        inputs: list[list[str]],
        *,
        batch_size: int = 32,
        show_progress_bar: bool | None = None,
        convert_to_numpy: bool = True,
    ) -> Sequence[float]:
        """Score query-document pairs."""
        ...


class RerankerProtocol(Protocol):
    """Behavior required from the graph's reranker node."""

    def rerank(
        self,
        query_text: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return chunks reranked for answer generation."""
        ...


class ScoreReranker:
    """Fast deterministic reranker that trusts fused retrieval scores."""

    def __init__(self, *, top_k: int = 5) -> None:
        """Configure how many fused retrieval results to keep."""
        if top_k < 1:
            msg = "top_k must be at least 1"
            raise ValueError(msg)
        self.top_k = top_k

    def rerank(
        self,
        query_text: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Return top chunks by current retrieval score."""
        del query_text
        limited_top_k = top_k or self.top_k
        ranked_chunks = sorted(chunks, key=lambda chunk: chunk.score, reverse=True)
        return [
            chunk.model_copy(update={"rank": rank})
            for rank, chunk in enumerate(ranked_chunks[:limited_top_k], start=1)
        ]


class LocalReranker:
    """Local cross-encoder reranker over retrieved CV chunks."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL,
        top_k: int = 5,
        max_input_chunks: int = 6,
        batch_size: int = 16,
        model_factory: Callable[[str], CrossEncoderProtocol] | None = None,
    ) -> None:
        """Load the configured cross-encoder once for repeated reranking calls."""
        if top_k < 1:
            msg = "top_k must be at least 1"
            raise ValueError(msg)
        if max_input_chunks < 1:
            msg = "max_input_chunks must be at least 1"
            raise ValueError(msg)
        self.top_k = top_k
        self.max_input_chunks = max_input_chunks
        self.batch_size = batch_size
        self._model_name = model_name
        self._model_factory = model_factory or _default_cross_encoder_factory
        self._model: CrossEncoderProtocol | None = None

    def rerank(
        self,
        query_text: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """Sort chunks by cross-encoder relevance score and rewrite ranks."""
        if not chunks:
            return []

        limited_top_k = top_k or self.top_k
        candidate_chunks = chunks[: self.max_input_chunks]
        sentence_pairs = [[query_text, chunk.text] for chunk in candidate_chunks]
        raw_scores = self._get_model().predict(
            sentence_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = [float(score) for score in raw_scores]
        ranked_pairs = sorted(
            zip(candidate_chunks, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        reranked_chunks: list[RetrievedChunk] = []
        for rank, (chunk, score) in enumerate(ranked_pairs[:limited_top_k], start=1):
            reranked_chunks.append(
                chunk.model_copy(update={"score": score, "rank": rank})
            )
        return reranked_chunks

    def _get_model(self) -> CrossEncoderProtocol:
        if self._model is None:
            self._model = self._model_factory(self._model_name)
        return self._model


def reranker_node(
    state: RAGState,
    *,
    reranker: RerankerProtocol,
    top_k: int | None = None,
) -> dict[str, list[RetrievedChunk]]:
    """LangGraph reranker node over retrieved chunks."""
    planner = state.get("planner")
    query_text = planner.primary_query if isinstance(planner, PlannerOutput) else None
    if query_text is None:
        user_query = state.get("user_query")
        if isinstance(user_query, str) and user_query.strip():
            query_text = user_query
    if query_text is None:
        msg = "reranker state must include planner output or user_query"
        raise TypeError(msg)

    chunks = state.get("retrieved_chunks")
    if chunks is None:
        msg = "reranker state must include retrieved chunks"
        raise TypeError(msg)

    return {
        "reranked_chunks": reranker.rerank(
            query_text,
            chunks,
            top_k=top_k,
        )
    }


def _default_cross_encoder_factory(model_name: str) -> CrossEncoderProtocol:
    module = import_module("sentence_transformers")
    cross_encoder = module.CrossEncoder
    return cast("CrossEncoderProtocol", cast("object", cross_encoder(model_name)))
