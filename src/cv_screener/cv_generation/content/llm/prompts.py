"""Prompt construction utilities for model-backed CV generation."""

import json
from typing import Literal

from openai.types.chat import ChatCompletionMessageParam

from cv_screener.cv_generation.content.schema import CVProfileDraft

CVLanguage = Literal["english", "spanish", "french"]
CVTargetPages = Literal[1, 2, 3]


_SCHEMA_JSON = json.dumps(
    CVProfileDraft.model_json_schema(),
    indent=2,
    ensure_ascii=False,
)

_PAGE_GUIDANCE: dict[CVTargetPages, str] = {
    1: (
        "Generate a concise one-page CV. "
        "Use 1-2 experience entries, 3-5 skills, and short descriptions."
    ),
    2: (
        "Generate a medium-length two-page CV. "
        "Use 2-4 experience entries, 6-10 skills, and moderately detailed descriptions."
    ),
    3: (
        "Generate a detailed three-page CV. "
        "Use 4-6 experience entries, 10-16 skills, and richer descriptions."
    ),
}


_SYSTEM_PROMPT = """\
You generate realistic but entirely fictional technical CV profiles.

Rules:
- Return exactly one valid JSON object.
- Return JSON only; no markdown fences or explanatory text.
- Do not use real people, real phone numbers, or real email addresses.
- Keep candidates varied in seniority, domain, education, location, and skills.
- All fields must satisfy the provided JSON schema.
- Use concise, realistic CV content.
"""


def build_generation_messages(
    *,
    language: CVLanguage = "english",
    target_pages: CVTargetPages = 1,
    last_error: str | None = None,
) -> list[ChatCompletionMessageParam]:
    """Build prompt messages for one CV generation call."""
    messages: list[ChatCompletionMessageParam] = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Generate one fictional technical CV profile in {language}.\n"
                f"{_PAGE_GUIDANCE[target_pages]}\n\n"
                "The generated content should be appropriate for the requested "
                f"{target_pages}-page CV when rendered as a PDF.\n\n"
                "Match this JSON schema:\n\n"
                f"{_SCHEMA_JSON}"
            ),
        },
    ]

    if last_error:
        messages.append(
            {
                "role": "user",
                "content": (
                    "The previous JSON response failed validation.\n"
                    "Validation error, delimited by triple backticks:\n"
                    f"```text\n{last_error}\n```\n"
                    "Return exactly one corrected JSON object."
                ),
            }
        )

    return messages
