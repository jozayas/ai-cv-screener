"""Hybrid retrieval module: dense + sparse search with RRF fusion."""

from cv_screener.retrieval.hybrid import HybridRetriever
from cv_screener.retrieval.schema import HybridRetrievalConfig, RetrievedChunk

__all__ = [
    "HybridRetrievalConfig",
    "HybridRetriever",
    "RetrievedChunk",
]
