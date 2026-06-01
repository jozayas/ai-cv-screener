"""Response parsing helpers for model-backed CV generation."""

import json
import re


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
