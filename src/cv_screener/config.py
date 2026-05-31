"""Environment-backed configuration for the CV screener."""

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class GenerationSettings(BaseSettings):
    """Settings for generating CV content with an OpenAI-compatible model."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_base_url: str = Field(default="http://localhost:11434/v1")
    openai_api_key: SecretStr = Field(default=SecretStr("ollama"))
    generation_model: str = Field(default="gemma3:12b")
    generation_temperature: float = Field(default=0.8, ge=0, le=2)
    generation_max_retries: int = Field(default=2, ge=0, le=5)
    generation_max_concurrency: int = Field(default=2, ge=1, le=16)
    generation_min_interval_seconds: float = Field(default=0.35, ge=0, le=30)
    generation_retry_base_delay_seconds: float = Field(default=0.5, ge=0.05, le=30)
    generation_retry_max_delay_seconds: float = Field(default=8.0, ge=0.1, le=120)
    image_generation_max_concurrency: int = Field(default=1, ge=1, le=8)


class QdrantSettings(BaseSettings):
    """Settings for the local Qdrant vector store."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    qdrant_url: str = Field(default="http://localhost:6333")
    qdrant_check_compatibility: bool = Field(default=False)


class RAGModelSettings(BaseSettings):
    """Settings for the RAG chat runtime over an OpenAI-compatible endpoint."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_base_url: str = Field(default="http://localhost:11434/v1")
    openai_api_key: SecretStr = Field(default=SecretStr("ollama"))
    rag_model: str = Field(default="gemma3:12b")
    rag_temperature: float = Field(default=0, ge=0, le=2)
    rag_max_retries: int = Field(default=2, ge=0, le=5)


class LangSmithSettings(BaseSettings):
    """Optional LangSmith tracing configuration for the RAG runtime."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    langsmith_tracing: bool = Field(default=False)
    langsmith_project: str = Field(default="cv-screener")
    langsmith_api_key: SecretStr | None = Field(default=None)
    langsmith_endpoint: str = Field(default="https://api.smith.langchain.com")


class SQLiteSettings(BaseSettings):
    """Settings for local SQLite persistence."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    sqlite_path: str = Field(default="data/cv_screener.db")
