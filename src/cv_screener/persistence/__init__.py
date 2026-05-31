"""SQLite persistence for canonical CV and chunk storage."""

from cv_screener.persistence.repository import (
    CanonicalStoreProtocol,
    SQLiteCanonicalRepository,
)

__all__ = ["CanonicalStoreProtocol", "SQLiteCanonicalRepository"]
