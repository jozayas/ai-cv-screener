"""Prompt construction for model-backed CV generation."""

import json

from openai.types.chat import ChatCompletionMessageParam

from cv_screener.cv_generation.content.schema import CVProfileDraft


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
