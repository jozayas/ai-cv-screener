"""Hybrid retrieval module: dense + sparse search with RRF fusion."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from cv_screener.retrieval.hybrid import HybridRetriever
    from cv_screener.retrieval.schema import HybridRetrievalConfig, RetrievedChunk

__all__ = [
    "HybridRetrievalConfig",
    "HybridRetriever",
    "RetrievedChunk",
]


def __getattr__(name: str) -> object:
    """Load retrieval exports on demand to keep package import cheap."""
    if name in {"HybridRetrievalConfig", "RetrievedChunk"}:
        schema = import_module("cv_screener.retrieval.schema")
        return getattr(schema, name)
    if name == "HybridRetriever":
        return import_module("cv_screener.retrieval.hybrid").HybridRetriever
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)
