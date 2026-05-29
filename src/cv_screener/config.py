"""Environment-backed configuration for the CV screener."""

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class GenerationSettings(BaseSettings):
    """Settings for generating CV content with an OpenAI-compatible model."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    openai_base_url: str = Field(default="http://localhost:11434/v1")
    openai_api_key: str = Field(default="ollama")
    generation_model: str = Field(default="gemma3:12b")
    generation_temperature: float = Field(default=0.8, ge=0, le=2)
    generation_max_retries: int = Field(default=2, ge=0, le=5)
