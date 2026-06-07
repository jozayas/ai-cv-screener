"""OpenAI-compatible transport for model-backed CV generation."""

from openai import OpenAI

from cv_screener.config import GenerationConfig
from cv_screener.cv_generation.content.llm.prompts import (
    CVLanguage,
    CVTargetPages,
    build_generation_messages,
)


class OpenAICVGenerationClient:
    """Thin client for requesting raw CV draft completions."""

    def __init__(self, settings: GenerationConfig) -> None:
        """Initialize the OpenAI-compatible client."""
        self.settings = settings
        self.client = OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key.get_secret_value(),
        )

    def request_draft(
        self,
        *,
        language: CVLanguage = "english",
        target_pages: CVTargetPages = 1,
        last_error: str | None = None,
    ) -> str:
        """Request a single raw draft completion from the configured model."""
        completion = self.client.chat.completions.create(
            model=self.settings.generation_model,
            temperature=self.settings.generation_temperature,
            messages=build_generation_messages(
                language=language,
                target_pages=target_pages,
                last_error=last_error,
            ),
        )
        content = completion.choices[0].message.content
        if not content:
            message = "model returned an empty completion"
            raise ValueError(message)
        return content
