"""Pydantic models for Qdrant indexing."""

from typing import Literal

from pydantic import BaseModel, Field

from cv_screener.config import QdrantSettings


class QdrantIndexConfig(BaseModel):
    """Configuration for semantic chunk indexing in Qdrant."""

    url: str = Field(
        default_factory=lambda: QdrantSettings().qdrant_url,
        min_length=1,
        description="Base URL for the Qdrant HTTP API.",
    )
    check_compatibility: bool = Field(
        default_factory=lambda: QdrantSettings().qdrant_check_compatibility,
        description="Whether the client should verify server compatibility at startup.",
    )
    collection_name: str = Field(
        default="cv_chunks",
        min_length=1,
        description="Target Qdrant collection for CV chunks.",
    )
    embedding_model_name: str = Field(
        default="BAAI/bge-small-en-v1.5",
        min_length=1,
        description="Embedding model used to generate chunk vectors.",
    )
    vector_size: int = Field(
        default=384,
        ge=1,
        description="Expected dense vector size for the configured embedding model.",
    )
    distance: Literal["cosine", "dot", "euclid", "manhattan"] = Field(
        default="cosine",
        description="Qdrant distance metric for semantic search.",
    )
    batch_size: int = Field(
        default=64,
        ge=1,
        description="Number of points per upload batch.",
    )
    parallel: int = Field(
        default=1,
        ge=1,
        description="Number of parallel upload workers for Qdrant client uploads.",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        description="Maximum number of retries for failed Qdrant upload batches.",
    )
    wait: bool = Field(
        default=True,
        description="Wait for Qdrant to apply writes before returning.",
    )
    sparse_vector_name: str = Field(
        default="bm25",
        min_length=1,
        description="Named sparse vector for BM25 keyword search in Qdrant.",
    )
    enable_bm25: bool = Field(
        default=True,
        description="Whether to build and upload BM25 sparse vectors alongside dense vectors.",
    )
    bm25_model_name: str = Field(
        default="Qdrant/bm25",
        min_length=1,
        description="FastEmbed BM25 model name for sparse vector encoding.",
    )
