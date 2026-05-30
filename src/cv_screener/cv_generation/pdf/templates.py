"""Template metadata and deterministic selection for CV PDFs."""

from collections.abc import Mapping
from dataclasses import dataclass

from cv_screener.cv_generation.content.schema import CVProfile
from cv_screener.cv_generation.pdf.sections import (
    SECTION_EDUCATION,
    SECTION_EXPERIENCE,
    SECTION_SKILLS,
    SECTION_SUMMARY,
)


@dataclass(frozen=True)
class TemplatePreset:
    """A predefined arrangement of CV sections for one template."""

    preset_id: str
    main_sections: tuple[str, ...]
    sidebar_sections: tuple[str, ...] = ()


@dataclass(frozen=True)
class TemplateDefinition:
    """Metadata for a fixed PDF template."""

    template_id: str
    template_name: str
    column_layout: str
    section_titles: Mapping[str, str]
    presets: tuple[TemplatePreset, ...]


@dataclass(frozen=True)
class TemplateSelection:
    """Deterministically selected template and preset for a CV."""

    template_id: str
    section_order_preset: str


TEMPLATE_DEFINITIONS: tuple[TemplateDefinition, ...] = (
    TemplateDefinition(
        template_id="moderncv_inspired",
        template_name="ModernCV Inspired",
        column_layout="single",
        section_titles={
            SECTION_SUMMARY: "Profile",
            SECTION_EXPERIENCE: "Professional Experience",
            SECTION_SKILLS: "Core Skills",
            SECTION_EDUCATION: "Education",
        },
        presets=(
            TemplatePreset(
                preset_id="summary-skills-experience-education",
                main_sections=(
                    SECTION_SUMMARY,
                    SECTION_SKILLS,
                    SECTION_EXPERIENCE,
                    SECTION_EDUCATION,
                ),
            ),
            TemplatePreset(
                preset_id="summary-experience-skills-education",
                main_sections=(
                    SECTION_SUMMARY,
                    SECTION_EXPERIENCE,
                    SECTION_SKILLS,
                    SECTION_EDUCATION,
                ),
            ),
            TemplatePreset(
                preset_id="experience-skills-education-summary",
                main_sections=(
                    SECTION_EXPERIENCE,
                    SECTION_SKILLS,
                    SECTION_EDUCATION,
                    SECTION_SUMMARY,
                ),
            ),
        ),
    ),
    TemplateDefinition(
        template_id="deedy_inspired",
        template_name="Deedy Inspired",
        column_layout="double",
        section_titles={
            SECTION_SUMMARY: "Summary",
            SECTION_EXPERIENCE: "Experience",
            SECTION_SKILLS: "Technical Skills",
            SECTION_EDUCATION: "Education",
        },
        presets=(
            TemplatePreset(
                preset_id="sidebar-skills-education_main-summary-experience",
                main_sections=(SECTION_SUMMARY, SECTION_EXPERIENCE),
                sidebar_sections=(SECTION_SKILLS, SECTION_EDUCATION),
            ),
            TemplatePreset(
                preset_id="sidebar-summary-skills_main-experience-education",
                main_sections=(SECTION_EXPERIENCE, SECTION_EDUCATION),
                sidebar_sections=(SECTION_SUMMARY, SECTION_SKILLS),
            ),
        ),
    ),
    TemplateDefinition(
        template_id="awesome_cv_inspired",
        template_name="Awesome-CV Inspired",
        column_layout="single",
        section_titles={
            SECTION_SUMMARY: "About",
            SECTION_EXPERIENCE: "Work Experience",
            SECTION_SKILLS: "Strengths",
            SECTION_EDUCATION: "Academic Background",
        },
        presets=(
            TemplatePreset(
                preset_id="experience-summary-skills-education",
                main_sections=(
                    SECTION_EXPERIENCE,
                    SECTION_SUMMARY,
                    SECTION_SKILLS,
                    SECTION_EDUCATION,
                ),
            ),
            TemplatePreset(
                preset_id="summary-experience-education-skills",
                main_sections=(
                    SECTION_SUMMARY,
                    SECTION_EXPERIENCE,
                    SECTION_EDUCATION,
                    SECTION_SKILLS,
                ),
            ),
        ),
    ),
    TemplateDefinition(
        template_id="altacv_inspired",
        template_name="AltaCV Inspired",
        column_layout="double",
        section_titles={
            SECTION_SUMMARY: "Profile",
            SECTION_EXPERIENCE: "Experience",
            SECTION_SKILLS: "Toolbox",
            SECTION_EDUCATION: "Studies",
        },
        presets=(
            TemplatePreset(
                preset_id="sidebar-summary-education_main-experience-skills",
                main_sections=(SECTION_EXPERIENCE, SECTION_SKILLS),
                sidebar_sections=(SECTION_SUMMARY, SECTION_EDUCATION),
            ),
            TemplatePreset(
                preset_id="sidebar-skills-summary_main-experience-education",
                main_sections=(SECTION_EXPERIENCE, SECTION_EDUCATION),
                sidebar_sections=(SECTION_SKILLS, SECTION_SUMMARY),
            ),
            TemplatePreset(
                preset_id="sidebar-education-skills_main-summary-experience",
                main_sections=(SECTION_SUMMARY, SECTION_EXPERIENCE),
                sidebar_sections=(SECTION_EDUCATION, SECTION_SKILLS),
            ),
        ),
    ),
)

def select_template(profile: CVProfile) -> tuple[TemplateDefinition, TemplatePreset]:
    """Deterministically select a template and section preset for a profile."""
    profile_key = profile.candidate_id.int
    definition = TEMPLATE_DEFINITIONS[profile_key % len(TEMPLATE_DEFINITIONS)]
    preset_index = (profile_key // len(TEMPLATE_DEFINITIONS)) % len(definition.presets)
    preset = definition.presets[preset_index]
    return definition, preset


def describe_template_selection(profile: CVProfile) -> TemplateSelection:
    """Return the deterministic template selection for a profile."""
    definition, preset = select_template(profile)
    return TemplateSelection(
        template_id=definition.template_id,
        section_order_preset=preset.preset_id,
    )
