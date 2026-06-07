from pathlib import Path

import pytest

from cv_screener.config import (
    AppSettings,
    GenerationConfig,
    ImageGenerationProvider,
    QdrantConfig,
    RAGConfig,
)
from cv_screener.ingestion.indexing.schema import QdrantIndexConfig


def test_qdrant_settings_default_to_localhost() -> None:
    settings = QdrantConfig()

    assert settings.qdrant_url == "http://localhost:6333"
    assert not settings.qdrant_check_compatibility


def test_qdrant_settings_read_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT__QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT__QDRANT_CHECK_COMPATIBILITY", "true")

    settings = AppSettings().qdrant

    assert settings.qdrant_url == "http://qdrant:6333"
    assert settings.qdrant_check_compatibility


def test_qdrant_index_config_defaults_from_qdrant_settings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("QDRANT__QDRANT_URL", "http://qdrant:6333")
    monkeypatch.setenv("QDRANT__QDRANT_CHECK_COMPATIBILITY", "true")

    config = QdrantIndexConfig()

    assert config.url == "http://qdrant:6333"
    assert config.check_compatibility


def test_rag_model_settings_default_to_local_ollama(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("RAG__OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("RAG__OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("RAG__RAG_MODEL", raising=False)
    monkeypatch.delenv("RAG__RAG_TEMPERATURE", raising=False)
    monkeypatch.delenv("RAG__RAG_MAX_RETRIES", raising=False)
    monkeypatch.delenv("RAG__RAG_ENABLE_LLM_REVIEW", raising=False)
    monkeypatch.delenv("RAG__RAG_ENABLE_CROSS_ENCODER_RERANK", raising=False)
    monkeypatch.delenv("RAG__RAG_MAX_RETRIEVAL_QUERIES", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = RAGConfig()

    assert settings.openai_base_url == "http://localhost:11434/v1"
    assert settings.openai_api_key.get_secret_value() == "ollama"
    assert settings.rag_model == "gemma3:12b"
    assert settings.rag_temperature == 0
    assert settings.rag_max_retries == 2
    assert not settings.rag_enable_llm_review
    assert settings.rag_enable_cross_encoder_rerank
    assert settings.rag_max_retrieval_queries == 1


def test_rag_model_settings_read_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAG__OPENAI_BASE_URL", "http://llm-gateway:4000/v1")
    monkeypatch.setenv("RAG__OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("RAG__RAG_MODEL", "qwen3:14b")
    monkeypatch.setenv("RAG__RAG_TEMPERATURE", "0.2")
    monkeypatch.setenv("RAG__RAG_MAX_RETRIES", "4")
    monkeypatch.setenv("RAG__RAG_ENABLE_LLM_REVIEW", "true")
    monkeypatch.setenv("RAG__RAG_ENABLE_CROSS_ENCODER_RERANK", "true")
    monkeypatch.setenv("RAG__RAG_MAX_RETRIEVAL_QUERIES", "3")

    settings = AppSettings().rag

    assert settings.openai_base_url == "http://llm-gateway:4000/v1"
    assert settings.openai_api_key.get_secret_value() == "test-key"
    assert settings.rag_model == "qwen3:14b"
    assert settings.rag_temperature == pytest.approx(0.2)
    assert settings.rag_max_retries == 4
    assert settings.rag_enable_llm_review
    assert settings.rag_enable_cross_encoder_rerank
    assert settings.rag_max_retrieval_queries == 3


def test_generation_settings_have_guardrail_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.delenv("GENERATION__GENERATION_MAX_CONCURRENCY", raising=False)
    monkeypatch.delenv("GENERATION__GENERATION_MIN_INTERVAL_SECONDS", raising=False)
    monkeypatch.delenv(
        "GENERATION__GENERATION_RETRY_BASE_DELAY_SECONDS",
        raising=False,
    )
    monkeypatch.delenv(
        "GENERATION__GENERATION_RETRY_MAX_DELAY_SECONDS",
        raising=False,
    )
    monkeypatch.delenv("GENERATION__IMAGE_GENERATION_PROVIDER", raising=False)
    monkeypatch.delenv("GENERATION__IMAGE_GENERATION_MAX_CONCURRENCY", raising=False)
    monkeypatch.delenv("GENERATION__IMAGE_GENERATION_MODEL", raising=False)
    monkeypatch.delenv("GENERATION__IMAGE_GENERATION_SIZE", raising=False)
    monkeypatch.delenv("GENERATION__HUGGINGFACE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)

    settings = GenerationConfig()

    assert settings.generation_max_concurrency == 2
    assert settings.generation_min_interval_seconds == pytest.approx(0.35)
    assert settings.generation_retry_base_delay_seconds == pytest.approx(0.5)
    assert settings.generation_retry_max_delay_seconds == pytest.approx(8.0)
    assert settings.image_generation_provider is ImageGenerationProvider.OPENAI
    assert settings.image_generation_max_concurrency == 1
    assert settings.image_generation_model == "gpt-image-1"
    assert settings.image_generation_size == "1024x1024"
    assert settings.huggingface_api_key is None


def test_generation_settings_read_huggingface_api_key_from_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GENERATION__HUGGINGFACE_API_KEY", "hf-secret")

    settings = AppSettings().generation

    assert settings.huggingface_api_key is not None
    assert settings.huggingface_api_key.get_secret_value() == "hf-secret"
