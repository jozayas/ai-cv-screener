"""Shared chunking constants and small helpers."""

import re
from collections.abc import Iterable

_HEADER_SECTION = "header"
_PERSON_LABEL = "person"
_SKILL_LABEL = "skill"
_COMPANY_LABEL = "company"
_UNIVERSITY_LABEL = "university"
_LINKEDIN_PATTERN = re.compile(
    r"https?://(?:[\w]+\.)?linkedin\.com/[^\s)]+",
    re.IGNORECASE,
)
_GITHUB_PATTERN = re.compile(
    r"https?://(?:[\w]+\.)?github\.com/[^\s)]+",
    re.IGNORECASE,
)
_SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n{2,}")


def dedupe(items: Iterable[str]) -> list[str]:
    """Keep first occurrence of each case-insensitive item."""
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        cleaned = item.strip()
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return ordered
