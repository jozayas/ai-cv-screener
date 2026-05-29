"""Draft CV content sources for local and model-backed generation."""

import json
from typing import Protocol

from loguru import logger
from pydantic import ValidationError

from cv_screener.config import GenerationSettings
from cv_screener.cv_generation.content.llm.client import OpenAICVGenerationClient
from cv_screener.cv_generation.content.llm.parsing import parse_json_payload
from cv_screener.cv_generation.content.schema import CVProfileDraft
from cv_screener.cv_generation.content.seed_data import SEED_PROFILES


class CVProfileSource(Protocol):
    """Profile source that can generate draft CV payloads."""

    def generate_draft(self, *, index: int) -> CVProfileDraft:
        """Generate a single draft CV payload."""
        ...


class SeededCVProfileSource:
    """Deterministic local source for development and tests."""

    def generate_draft(self, *, index: int) -> CVProfileDraft:
        """Generate a draft from the local seeded payloads."""
        logger.debug(
            "Using seeded CV draft source",
            candidate_index=index + 1,
            source="seeded",
        )
        return CVProfileDraft.model_validate(SEED_PROFILES[index % len(SEED_PROFILES)])


class OpenAICVProfileSource:
    """OpenAI-compatible source for model-backed CV generation."""

    def __init__(self, settings: GenerationSettings) -> None:
        """Initialize the OpenAI-compatible client."""
        self.settings = settings
        self.client = OpenAICVGenerationClient(settings)

    def generate_draft(self, *, index: int) -> CVProfileDraft:
        """Generate a draft CV payload from the configured model."""
        last_error: str | None = None
        for attempt in range(self.settings.generation_max_retries + 1):
            logger.debug(
                "Requesting CV draft from model",
                candidate_index=index + 1,
                progress_attempt=f"{attempt + 1}/{self.settings.generation_max_retries + 1}",
                attempt=attempt + 1,
                max_attempts=self.settings.generation_max_retries + 1,
                model=self.settings.generation_model,
            )
            content = self.client.request_draft(last_error=last_error)
            try:
                payload = parse_json_payload(content)
                logger.debug(
                    "Received valid CV draft from model",
                    candidate_index=index + 1,
                    progress_attempt=f"{attempt + 1}/{self.settings.generation_max_retries + 1}",
                    attempt=attempt + 1,
                    content_length=len(content),
                )
                return CVProfileDraft.model_validate(payload)
            except (json.JSONDecodeError, ValidationError) as error:
                last_error = str(error)
                logger.warning(
                    "Model response failed CV draft parsing or validation",
                    candidate_index=index + 1,
                    progress_attempt=f"{attempt + 1}/{self.settings.generation_max_retries + 1}",
                    attempt=attempt + 1,
                    max_attempts=self.settings.generation_max_retries + 1,
                    model=self.settings.generation_model,
                    content_length=len(content),
                    error_type=type(error).__name__,
                    error_message=last_error,
                )
        message = f"model failed to produce a valid CV draft after retries: {last_error}"
        raise ValueError(message)
