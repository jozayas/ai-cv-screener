"""Draft CV content sources for local and model-backed generation."""

from __future__ import annotations

import json
import random
from threading import Lock
from time import monotonic, sleep
from typing import TYPE_CHECKING, Protocol

from loguru import logger
from openai import APIConnectionError, APIError, APITimeoutError, RateLimitError
from pydantic import ValidationError

from cv_screener.cv_generation.content.llm.client import OpenAICVGenerationClient
from cv_screener.cv_generation.content.llm.parsing import parse_json_payload
from cv_screener.cv_generation.content.schema import CVProfileDraft
from cv_screener.cv_generation.content.seed_data import SEED_PROFILES

if TYPE_CHECKING:
    from cv_screener.config import GenerationConfig
    from cv_screener.cv_generation.content.llm.prompts import (
        CVLanguage,
        CVTargetPages,
    )


class CVProfileSource(Protocol):
    """Profile source that can generate draft CV payloads."""

    @property
    def max_concurrency(self) -> int:
        """Maximum safe parallelism for this source."""
        ...

    def generate_draft(self, *, index: int) -> CVProfileDraft:
        """Generate a single draft CV payload."""
        ...


class SeededCVProfileSource:
    """Deterministic local source for development and tests."""

    @property
    def max_concurrency(self) -> int:
        """Return safe local parallelism for seeded generation."""
        return 8

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

    def __init__(self, settings: GenerationConfig) -> None:
        """Initialize the OpenAI-compatible client."""
        self.settings = settings
        self.client = OpenAICVGenerationClient(settings)
        self._request_lock = Lock()
        self._last_request_at = 0.0

    @property
    def max_concurrency(self) -> int:
        """Return guarded concurrency for model-backed generation."""
        return self.settings.generation_max_concurrency

    def generate_draft(self, *, index: int) -> CVProfileDraft:
        """Generate a draft CV payload from the configured model."""
        language: CVLanguage = random.choice(["english", "spanish", "french"])  # noqa: S311
        target_pages: CVTargetPages = random.choice([1, 2, 3])  # noqa: S311
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
            content: str | None = None
            try:
                self._wait_for_rate_limit_slot()
                content = self.client.request_draft(
                    language=language,
                    target_pages=target_pages,
                    last_error=last_error,
                )
                payload = parse_json_payload(content)
                logger.debug(
                    "Received valid CV draft from model",
                    candidate_index=index + 1,
                    progress_attempt=f"{attempt + 1}/{self.settings.generation_max_retries + 1}",
                    attempt=attempt + 1,
                    content_length=len(content),
                )
                return CVProfileDraft.model_validate(payload)
            except (
                json.JSONDecodeError,
                ValidationError,
                APIError,
                RateLimitError,
                APIConnectionError,
                APITimeoutError,
            ) as error:
                last_error = str(error)
                logger.warning(
                    "Model response failed CV draft parsing or validation",
                    candidate_index=index + 1,
                    progress_attempt=f"{attempt + 1}/{self.settings.generation_max_retries + 1}",
                    attempt=attempt + 1,
                    max_attempts=self.settings.generation_max_retries + 1,
                    model=self.settings.generation_model,
                    content_length=len(content) if content is not None else 0,
                    error_type=type(error).__name__,
                    error_message=last_error,
                )
                if attempt < self.settings.generation_max_retries:
                    sleep(self._retry_delay_seconds(attempt))
        message = (
            f"model failed to produce a valid CV draft after retries: {last_error}"
        )
        raise ValueError(message)

    def _wait_for_rate_limit_slot(self) -> None:
        interval = self.settings.generation_min_interval_seconds
        if interval <= 0:
            return
        with self._request_lock:
            now = monotonic()
            elapsed = now - self._last_request_at
            if elapsed < interval:
                sleep(interval - elapsed)
            self._last_request_at = monotonic()

    def _retry_delay_seconds(self, attempt: int) -> float:
        base = self.settings.generation_retry_base_delay_seconds
        cap = self.settings.generation_retry_max_delay_seconds
        backoff = min(cap, base * (2**attempt))
        jitter = max(0.001, backoff * 0.1) * ((attempt % 3) + 1) / 3
        return backoff + jitter
