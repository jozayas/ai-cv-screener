"""Render-ready section payloads for CV PDFs."""

from collections.abc import Mapping
from typing import Any

from cv_screener.cv_generation.content.generator import slugify
from cv_screener.cv_generation.content.schema import CVProfile

SECTION_SUMMARY = "summary"
SECTION_EXPERIENCE = "experience"
SECTION_SKILLS = "skills"
SECTION_EDUCATION = "education"


def build_pdf_filename(profile: CVProfile) -> str:
    """Build the output PDF filename for a profile."""
    return f"{slugify(profile.full_name)}-{profile.candidate_id}.pdf"


def build_section_content(
    profile: CVProfile,
    section_titles: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Build render-ready section payloads from a CV profile."""
    return {
        SECTION_SUMMARY: {
            "id": SECTION_SUMMARY,
            "title": section_titles[SECTION_SUMMARY],
            "kind": SECTION_SUMMARY,
            "content": profile.professional_summary,
        },
        SECTION_EXPERIENCE: {
            "id": SECTION_EXPERIENCE,
            "title": section_titles[SECTION_EXPERIENCE],
            "kind": SECTION_EXPERIENCE,
            "items": list(profile.experience),
        },
        SECTION_SKILLS: {
            "id": SECTION_SKILLS,
            "title": section_titles[SECTION_SKILLS],
            "kind": SECTION_SKILLS,
            "items": list(profile.skills),
        },
        SECTION_EDUCATION: {
            "id": SECTION_EDUCATION,
            "title": section_titles[SECTION_EDUCATION],
            "kind": SECTION_EDUCATION,
            "items": list(profile.education),
        },
    }
