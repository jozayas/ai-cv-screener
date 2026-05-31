"""Cross-encoder reranking for retrieved CV chunks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from sentence_transformers import CrossEncoder

from cv_screener.rag.schema import PlannerOutput

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from cv_screener.rag.state import RAGState
    from cv_screener.retrieval.schema import RetrievedChunk


DEFAULT_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
FALLBACK_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


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


class LocalReranker:
    """Local cross-encoder reranker over retrieved CV chunks."""

    def __init__(
        self,
        *,
        model_name: str = DEFAULT_RERANKER_MODEL,
        top_k: int = 5,
        batch_size: int = 16,
        model_factory: Callable[[str], CrossEncoderProtocol] | None = None,
    ) -> None:
        """Load the configured cross-encoder once for repeated reranking calls."""
        self.top_k = top_k
        self.batch_size = batch_size
        factory = model_factory or _default_cross_encoder_factory
        self._model = factory(model_name)

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
        sentence_pairs = [[query_text, chunk.text] for chunk in chunks]
        raw_scores = self._model.predict(
            sentence_pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = [float(score) for score in raw_scores]
        ranked_pairs = sorted(
            zip(chunks, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        reranked_chunks: list[RetrievedChunk] = []
        for rank, (chunk, score) in enumerate(ranked_pairs[:limited_top_k], start=1):
            reranked_chunks.append(
                chunk.model_copy(update={"score": score, "rank": rank})
            )
        return reranked_chunks


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
    return cast("CrossEncoderProtocol", cast("object", CrossEncoder(model_name)))
