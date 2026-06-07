"""Environment-backed configuration for the CV screener."""

from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class ImageGenerationProvider(StrEnum):
    """Supported image-generation backends."""

    OPENAI = "openai"
    POLLINATIONS = "pollinations"
    HUGGINGFACE = "huggingface"


class PathsSettings(BaseModel):
    """Filesystem paths used by local commands and services."""

    model_config = ConfigDict(extra="forbid")

    cv_content_dir: Path = Path("data/cvs_contents")
    cv_pdf_dir: Path = Path("data/cv_pdfs")
    generated_photo_dir: Path = Path("data/generated/photos")


class ServeSettings(BaseModel):
    """Defaults for serving the local Chainlit host app."""

    model_config = ConfigDict(extra="forbid")

    host: str = "127.0.0.1"
    port: int = Field(default=8000, ge=1, le=65535)
    headless: bool = True
    watch: bool = False


class GenerationConfig(BaseModel):
    """Generation and image backend settings."""

    model_config = ConfigDict(extra="forbid")

    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: SecretStr = SecretStr("ollama")
    generation_model: str = "gemma3:12b"
    generation_temperature: float = Field(default=0.8, ge=0, le=2)
    generation_max_retries: int = Field(default=2, ge=0, le=5)
    generation_max_concurrency: int = Field(default=2, ge=1, le=16)
    generation_min_interval_seconds: float = Field(default=0.35, ge=0, le=30)
    generation_retry_base_delay_seconds: float = Field(default=0.5, ge=0.05, le=30)
    generation_retry_max_delay_seconds: float = Field(default=8.0, ge=0.1, le=120)
    image_generation_provider: ImageGenerationProvider = ImageGenerationProvider.OPENAI
    image_generation_max_concurrency: int = Field(default=1, ge=1, le=8)
    image_generation_model: str = "gpt-image-1"
    image_generation_size: str = "1024x1024"
    pollinations_base_url: str = "https://image.pollinations.ai/prompt"
    pollinations_model: str = "flux"
    huggingface_base_url: str = "https://api-inference.huggingface.co/models"
    huggingface_model: str = "stabilityai/stable-diffusion-xl-base-1.0"
    huggingface_api_key: SecretStr | None = None


class QdrantConfig(BaseModel):
    """Qdrant connection settings."""

    model_config = ConfigDict(extra="forbid")

    qdrant_url: str = "http://localhost:6333"
    qdrant_check_compatibility: bool = False


class RAGConfig(BaseModel):
    """RAG runtime model settings."""

    model_config = ConfigDict(extra="forbid")

    openai_base_url: str = "http://localhost:11434/v1"
    openai_api_key: SecretStr = SecretStr("ollama")
    rag_model: str = "gemma3:12b"
    rag_temperature: float = Field(default=0, ge=0, le=2)
    rag_max_retries: int = Field(default=2, ge=0, le=5)
    rag_enable_llm_review: bool = False
    rag_enable_cross_encoder_rerank: bool = True
    rag_max_retrieval_queries: int = Field(default=1, ge=1, le=3)


class SQLiteConfig(BaseModel):
    """SQLite persistence settings."""

    model_config = ConfigDict(extra="forbid")

    sqlite_path: str = "data/cv_screener.db"


class LookupConfig(BaseModel):
    """Deterministic lookup settings."""

    model_config = ConfigDict(extra="forbid")

    candidate_name_min_score: float = Field(default=0.72, ge=0, le=1)


class AppSettings(BaseSettings):
    """Top-level application settings for explicit runtime composition."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_nested_delimiter="__",
        nested_model_default_partial_update=True,
    )

    paths: PathsSettings = Field(default_factory=PathsSettings)
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    qdrant: QdrantConfig = Field(default_factory=QdrantConfig)
    rag: RAGConfig = Field(default_factory=RAGConfig)
    sqlite: SQLiteConfig = Field(default_factory=SQLiteConfig)
    lookup: LookupConfig = Field(default_factory=LookupConfig)
    serve: ServeSettings = Field(default_factory=ServeSettings)
