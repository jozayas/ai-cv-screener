"""Hybrid (dense + sparse) retriever over indexed CV chunks."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from fastembed import TextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import Fusion, FusionQuery, Prefetch, SparseVector

from cv_screener.ingestion.indexing.bm25 import BM25Encoder, BM25EncoderProtocol
from cv_screener.retrieval.schema import HybridRetrievalConfig, RetrievedChunk

if TYPE_CHECKING:
    from cv_screener.ingestion.indexing.protocols import (
        EmbeddingModel,
        QdrantQueryClientProtocol,
    )


class HybridRetriever:
    """Retrieve chunks with hybrid dense + sparse search and RRF fusion."""

    config: HybridRetrievalConfig
    _client: QdrantQueryClientProtocol
    _embedding_model: EmbeddingModel
    _bm25_encoder: BM25EncoderProtocol | None

    def __init__(
        self,
        config: HybridRetrievalConfig | None = None,
        *,
        client: QdrantQueryClientProtocol | None = None,
        embedding_model: EmbeddingModel | None = None,
        bm25_encoder: BM25EncoderProtocol | None = None,
    ) -> None:
        """Bind Qdrant and embedding dependencies to the retrieval config."""
        self.config = config or HybridRetrievalConfig()
        if client is not None:
            self._client = client
        else:
            qdrant_client = QdrantClient(
                url=self.config.url,
                check_compatibility=False,
            )
            self._client = cast(
                "QdrantQueryClientProtocol",
                cast("object", qdrant_client),
            )
        self._embedding_model = embedding_model or TextEmbedding(
            model_name=self.config.embedding_model_name,
        )
        if bm25_encoder is not None:
            self._bm25_encoder = bm25_encoder
        elif self.config.enable_bm25:
            self._bm25_encoder = BM25Encoder(model_name=self.config.bm25_model_name)
        else:
            self._bm25_encoder = None

    def retrieve(self, query_text: str) -> list[RetrievedChunk]:
        """Run hybrid retrieval and return ranked chunks."""
        dense_vector = next(iter(self._embedding_model.embed([query_text])))
        dense_list = list(dense_vector)

        prefetch: list[Prefetch] = [
            Prefetch(
                query=dense_list,
                using=self.config.dense_vector_name,
                limit=self.config.semantic_top_k,
            ),
        ]

        if self._bm25_encoder is not None:
            sparse_dict = self._bm25_encoder.encode([query_text])[0]
            prefetch.append(
                Prefetch(
                    query=SparseVector(
                        indices=list(sparse_dict.keys()),
                        values=list(sparse_dict.values()),
                    ),
                    using=self.config.sparse_vector_name,
                    limit=self.config.bm25_top_k,
                ),
            )

        response = self._client.query_points(
            collection_name=self.config.collection_name,
            prefetch=prefetch,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=self.config.fusion_top_k,
            with_payload=True,
            with_vectors=False,
        )

        chunks: list[RetrievedChunk] = []
        for rank, point in enumerate(response.points, start=1):
            payload = point.payload or {}
            chunks.append(
                RetrievedChunk(
                    candidate_name=payload.get("candidate_name"),
                    source_file=payload.get("source_file", ""),
                    document_title=payload.get("document_title", ""),
                    page=payload.get("page", 1),
                    section=payload.get("section", ""),
                    text=payload.get("text", ""),
                    score=point.score,
                    rank=rank,
                )
            )
        return chunks
