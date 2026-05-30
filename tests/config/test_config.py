from pathlib import Path

import pytest

from cv_screener.config import LangSmithSettings, QdrantSettings, RAGModelSettings
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


def test_rag_model_settings_default_to_local_ollama(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RAG_MODEL", raising=False)
    monkeypatch.delenv("RAG_TEMPERATURE", raising=False)
    monkeypatch.delenv("RAG_MAX_RETRIES", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = RAGModelSettings()

    assert settings.openai_base_url == "http://localhost:11434/v1"
    assert settings.openai_api_key.get_secret_value() == "ollama"
    assert settings.rag_model == "gemma3:12b"
    assert settings.rag_temperature == 0
    assert settings.rag_max_retries == 2


def test_rag_model_settings_read_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPENAI_BASE_URL", "http://llm-gateway:4000/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RAG_MODEL", "qwen3:14b")
    monkeypatch.setenv("RAG_TEMPERATURE", "0.2")
    monkeypatch.setenv("RAG_MAX_RETRIES", "4")

    settings = RAGModelSettings()

    assert settings.openai_base_url == "http://llm-gateway:4000/v1"
    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert settings.rag_model == "qwen3:14b"
    assert settings.rag_temperature == pytest.approx(0.2)
    assert settings.rag_max_retries == 4


def test_langsmith_settings_default_to_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGSMITH_PROJECT", raising=False)
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGSMITH_ENDPOINT", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = LangSmithSettings()

    assert not settings.langsmith_tracing
    assert settings.langsmith_project == "cv-screener"
    assert settings.langsmith_api_key is None
    assert settings.langsmith_endpoint == "https://api.smith.langchain.com"


def test_langsmith_settings_read_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "cv-screener-demo")
    monkeypatch.setenv("LANGSMITH_API_KEY", "langsmith-key")
    monkeypatch.setenv("LANGSMITH_ENDPOINT", "https://smith.internal")

    settings = LangSmithSettings()

    assert settings.langsmith_tracing
    assert settings.langsmith_project == "cv-screener-demo"
    assert settings.langsmith_api_key is not None
    assert settings.langsmith_api_key.get_secret_value() == "langsmith-key"
    assert settings.langsmith_endpoint == "https://smith.internal"
