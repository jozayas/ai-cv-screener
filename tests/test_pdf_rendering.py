from pathlib import Path
from uuid import UUID

from cv_screener.cv_generation.content.schema import CVProfile
from cv_screener.cv_generation.content.yaml_io import write_cv_profile
from cv_screener.cv_generation.pdf.renderer import (
    CVPDFRenderer,
    PDFRenderingService,
)
from cv_screener.cv_generation.pdf.sections import build_pdf_filename
from cv_screener.cv_generation.pdf.templates import describe_template_selection


def build_profile(candidate_id: str) -> CVProfile:
    return CVProfile.model_validate(
        {
            "candidate_id": candidate_id,
            "full_name": "Alex Example",
            "email": "alex@example.com",
            "phone": "+34 600 000 000",
            "location": "Madrid, Spain",
            "professional_summary": (
                "Backend engineer focused on Python, distributed systems, and delivery."
            ),
            "skills": ["Python", "FastAPI", "Docker", "PostgreSQL"],
            "experience": [
                {
                    "company": "Example Corp",
                    "role": "Senior Backend Engineer",
                    "start_date": "2021-01-01",
                    "end_date": None,
                    "summary": "Built and operated internal APIs for customer workflows.",
                    "highlights": [
                        "Reduced API latency by 30%.",
                        "Migrated services to container-based deployment.",
                    ],
                    "technologies": ["Python", "FastAPI", "Docker"],
                }
            ],
            "education": [
                {
                    "institution": "Example University",
                    "degree": "BSc",
                    "field_of_study": "Computer Science",
                    "graduation_year": 2019,
                }
            ],
        }
    )


def test_build_pdf_filename_uses_profile_identity() -> None:
    profile = build_profile("00000000-0000-0000-0000-000000000001")

    filename = build_pdf_filename(profile)

    assert filename == "alex-example-00000000-0000-0000-0000-000000000001.pdf"


def test_template_selection_is_deterministic_by_candidate_id() -> None:
    profile = build_profile("00000000-0000-0000-0000-000000000005")

    selection = describe_template_selection(profile)

    assert selection.template_id == "deedy_inspired"
    assert selection.section_order_preset == "sidebar-summary-skills_main-experience-education"


def test_render_html_uses_selected_template_and_section_order() -> None:
    renderer = CVPDFRenderer()
    profile = build_profile("00000000-0000-0000-0000-000000000002")

    html = renderer.render_html(profile)

    summary_position = html.index("About")
    experience_position = html.index("Work Experience")
    education_position = html.index("Academic Background")
    skills_position = html.index("Strengths")
    assert "Awesome-CV Inspired" in html
    assert "data:image/svg+xml" not in html
    assert experience_position < summary_position < skills_position
    assert summary_position < education_position


def test_render_directory_writes_pdfs_for_yaml_profiles(tmp_path: Path) -> None:
    input_dir = tmp_path / "yaml"
    output_dir = tmp_path / "pdf"
    profile = build_profile("00000000-0000-0000-0000-000000000003")
    write_cv_profile(input_dir / "candidate.yaml", profile)

    class FakeRenderer(CVPDFRenderer):
        def write_pdf(self, profile: CVProfile, output_path: Path) -> None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_bytes(f"PDF:{profile.candidate_id}".encode())

    service = PDFRenderingService(
        input_dir=input_dir,
        output_dir=output_dir,
        renderer=FakeRenderer(),
    )

    rendered_files = service.render_directory()

    assert rendered_files == [
        output_dir / "alex-example-00000000-0000-0000-0000-000000000003.pdf"
    ]
    assert rendered_files[0].read_bytes() == b"PDF:00000000-0000-0000-0000-000000000003"


def test_template_selection_covers_all_fixed_templates() -> None:
    selections = {
        describe_template_selection(build_profile(str(UUID(int=index)))).template_id
        for index in range(1, 9)
    }

    assert selections == {
        "altacv_inspired",
        "awesome_cv_inspired",
        "deedy_inspired",
        "moderncv_inspired",
    }
