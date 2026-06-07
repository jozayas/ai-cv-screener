"""Pydantic models for hybrid retrieval."""

from pydantic import BaseModel, Field

from cv_screener.config import AppSettings


class HybridRetrievalConfig(BaseModel):
    """Configuration for hybrid (dense + sparse) retrieval."""

    url: str = Field(
        default_factory=lambda: AppSettings().qdrant.qdrant_url,
        min_length=1,
        description="Base URL for the Qdrant HTTP API.",
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
    dense_vector_name: str = Field(
        default="dense",
        min_length=1,
        description="Named dense vector for semantic search.",
    )
    sparse_vector_name: str = Field(
        default="bm25",
        min_length=1,
        description="Named sparse vector for BM25 keyword search.",
    )
    semantic_top_k: int = Field(
        default=12,
        ge=1,
        description="Number of results from the dense semantic prefetch.",
    )
    bm25_top_k: int = Field(
        default=12,
        ge=1,
        description="Number of results from the BM25 sparse prefetch.",
    )
    fusion_top_k: int = Field(
        default=10,
        ge=1,
        description="Number of results after RRF fusion.",
    )
    enable_bm25: bool = Field(
        default=True,
        description="Whether to include BM25 sparse search alongside dense search.",
    )
    bm25_model_name: str = Field(
        default="Qdrant/bm25",
        min_length=1,
        description="FastEmbed BM25 model name for sparse vector encoding.",
    )


class RetrievedChunk(BaseModel):
    """A single retrieved chunk with score and source metadata."""

    candidate_name: str | None = Field(
        default=None,
        description="Candidate display name from the chunk payload.",
    )
    source_file: str = Field(
        min_length=1,
        description="Source PDF filename for citations.",
    )
    document_title: str = Field(
        default="",
        description="PDF document title for citation display.",
    )
    page: int = Field(ge=1, description="1-indexed page number from the chunk.")
    section: str = Field(
        min_length=1,
        description="Section heading the chunk belongs to.",
    )
    text: str = Field(min_length=1, description="Chunk text content.")
    score: float = Field(description="Relevance score from the retrieval step.")
    rank: int = Field(ge=1, description="1-based rank after fusion.")
