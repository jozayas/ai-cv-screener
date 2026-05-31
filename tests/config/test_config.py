import pytest

from cv_screener.config import QdrantSettings
from cv_screener.ingestion.indexing.schema import QdrantIndexConfig


def test_qdrant_settings_default_to_localhost() -> None:
    settings = QdrantSettings()

    assert settings.qdrant_url == "http://localhost:6333"
    assert not settings.qdrant_check_compatibility


def test_qdrant_settings_read_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_CHECK_COMPATIBILITY", "true")

    settings = QdrantSettings()

    assert settings.qdrant_url == "http://qdrant:6333"
    assert settings.qdrant_check_compatibility


def test_qdrant_index_config_defaults_from_qdrant_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT_CHECK_COMPATIBILITY", "true")

    config = QdrantIndexConfig()

    assert config.url == "http://qdrant:6333"
    assert config.check_compatibility
