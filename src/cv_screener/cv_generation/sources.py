"""Draft CV content sources for local and model-backed generation."""

import json
import re
from typing import Protocol

from loguru import logger
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import ValidationError

from cv_screener.config import GenerationSettings
from cv_screener.cv_generation.schema import CVProfileDraft
from cv_screener.cv_generation.seed_data import SEED_PROFILES


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


def parse_json_payload(content: str) -> dict:
    """Parse a best-effort JSON object from a model response."""
    normalized = content.strip()
    if normalized.startswith("```"):
        normalized = normalized.strip("`")
        normalized = normalized.removeprefix("json").strip()
    json_start = normalized.find("{")
    json_end = normalized.rfind("}")
    if json_start != -1 and json_end != -1:
        normalized = normalized[json_start : json_end + 1]
    normalized = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", " ", normalized)
    normalized = normalized.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    return json.loads(normalized)


def build_generation_messages(
    *,
    last_error: str | None = None,
) -> list[ChatCompletionMessageParam]:
    """Build the prompt messages for a single CV generation call."""
    messages: list[ChatCompletionMessageParam] = [
        {
            "role": "system",
            "content": (
                "You generate realistic fake technical CVs. "
                "Return JSON only. Do not include markdown fences. "
                "Keep candidates varied in seniority, domain, education, and skills. "
                "Allow the CV content to be written in different natural languages. "
                "Do not include analysis, reasoning, or explanatory text. "
                "All string values must be valid JSON strings on a single line."
            ),
        },
        {
            "role": "user",
            "content": (
                "Generate one fake technical CV as JSON that matches this schema:\n"
                f"{json.dumps(CVProfileDraft.model_json_schema(), indent=2)}"
            ),
        },
    ]
    if last_error is not None:
        messages.append(
            {
                "role": "user",
                "content": (
                    "The previous response was invalid. "
                    f"Validation or parse error: {last_error}. "
                    "Return only one corrected JSON object."
                ),
            }
        )
    return messages


class OpenAICVProfileSource:
    """OpenAI-compatible source for model-backed CV generation."""

    def __init__(self, settings: GenerationSettings) -> None:
        """Initialize the OpenAI-compatible client."""
        self.settings = settings
        self.client = OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
        )

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
            completion = self.client.chat.completions.create(
                model=self.settings.generation_model,
                temperature=self.settings.generation_temperature,
                messages=build_generation_messages(last_error=last_error),
            )
            content = completion.choices[0].message.content
            if not content:
                message = "model returned an empty completion"
                raise ValueError(message)
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
