import base64
from pathlib import Path
from typing import Self
from uuid import UUID, uuid4

import pytest

from cv_screener.config import GenerationConfig, ImageGenerationProvider
from cv_screener.cv_generation.content.schema import CVProfile
from cv_screener.cv_generation.content.yaml_io import load_cv_profile, write_cv_profile
from cv_screener.cv_generation.photos.service import (
    CVPhotoGenerationService,
    HuggingFaceHeadshotGenerationClient,
    PollinationsHeadshotGenerationClient,
)

SAMPLE_PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO5X"
    "G9QAAAAASUVORK5CYII="
)


def _profile(candidate_id: UUID, *, photo_path: str | None = None) -> CVProfile:
    return CVProfile.model_validate(
        {
            "candidate_id": str(candidate_id),
            "full_name": "Marta Alvarez",
            "email": "marta.alvarez@example.com",
            "phone": "+34 600 111 111",
            "location": "Barcelona, Spain",
            "professional_summary": "Backend engineer with experience building APIs.",
            "skills": ["Python", "FastAPI"],
            "experience": [
                {
                    "company": "NovaStack",
                    "role": "Backend Engineer",
                    "start_date": "2021-03-01",
                    "end_date": None,
                    "summary": "Built backend services.",
                    "highlights": [],
                    "technologies": ["Python"],
                }
            ],
            "education": [
                {
                    "institution": "Universitat Politecnica de Catalunya",
                    "degree": "BSc",
                    "field_of_study": "Computer Engineering",
                    "graduation_year": 2020,
                }
            ],
            "photo_path": photo_path,
        }
    )


def test_generate_files_writes_photo_and_updates_yaml(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    content_dir = tmp_path / "content"
    photo_dir = tmp_path / "data" / "generated" / "photos"
    content_dir.mkdir()

    candidate_id = uuid4()
    profile_path = content_dir / "candidate.yaml"
    write_cv_profile(profile_path, _profile(candidate_id))

    prompts: list[str] = []

    class FakeClient:
        def generate_headshot(self, *, prompt: str) -> bytes:
            prompts.append(prompt)
            return SAMPLE_PNG_BYTES

    service = CVPhotoGenerationService(photo_dir=photo_dir, client=FakeClient())
    summary = service.generate_files([profile_path])

    expected_photo_path = Path("data/generated/photos") / f"{candidate_id}.png"
    updated_profile = load_cv_profile(profile_path)

    assert summary.profile_count == 1
    assert summary.generated_count == 1
    assert summary.skipped_count == 0
    assert expected_photo_path.exists()
    assert updated_profile.photo_path == expected_photo_path.as_posix()
    assert "Marta Alvarez" in prompts[0]
    assert "Barcelona, Spain" in prompts[0]


def test_generate_files_skips_profiles_with_valid_photo_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    content_dir = tmp_path / "content"
    photo_dir = tmp_path / "data" / "generated" / "photos"
    content_dir.mkdir()
    photo_dir.mkdir(parents=True)

    candidate_id = uuid4()
    expected_photo_path = Path("data/generated/photos") / f"{candidate_id}.png"
    (tmp_path / expected_photo_path).write_bytes(SAMPLE_PNG_BYTES)
    profile_path = content_dir / "candidate.yaml"
    write_cv_profile(
        profile_path,
        _profile(candidate_id, photo_path=expected_photo_path.as_posix()),
    )

    class FailingClient:
        def generate_headshot(self, *, prompt: str) -> bytes:
            _ = prompt
            message = "should not generate a new photo when one already exists"
            raise AssertionError(message)

    service = CVPhotoGenerationService(photo_dir=photo_dir, client=FailingClient())
    summary = service.generate_files([profile_path])

    updated_profile = load_cv_profile(profile_path)

    assert summary.profile_count == 1
    assert summary.generated_count == 0
    assert summary.skipped_count == 1
    assert updated_profile.photo_path == expected_photo_path.as_posix()


def test_service_uses_pollinations_client_when_configured(tmp_path: Path) -> None:
    service = CVPhotoGenerationService(
        photo_dir=tmp_path,
        settings=GenerationConfig(
            image_generation_provider=ImageGenerationProvider.POLLINATIONS,
        ),
    )

    assert isinstance(service.client, PollinationsHeadshotGenerationClient)


def test_service_uses_huggingface_client_when_configured(tmp_path: Path) -> None:
    service = CVPhotoGenerationService(
        photo_dir=tmp_path,
        settings=GenerationConfig(
            image_generation_provider=ImageGenerationProvider.HUGGINGFACE,
            huggingface_api_key="hf-secret",
        ),
    )

    assert isinstance(service.client, HuggingFaceHeadshotGenerationClient)


def test_pollinations_client_builds_image_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeResponse:
        content = SAMPLE_PNG_BYTES

        def raise_for_status(self) -> None:
            return None

        def read(self) -> bytes:
            return SAMPLE_PNG_BYTES

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: object,
            exc: object,
            tb: object,
        ) -> None:
            _ = (exc_type, exc, tb)

    def fake_get(
        url: str,
        *,
        params: dict[str, object],
        headers: dict[str, str],
        timeout: int,
    ) -> FakeResponse:
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        "cv_screener.cv_generation.photos.service.httpx.get",
        fake_get,
    )

    client = PollinationsHeadshotGenerationClient(GenerationConfig())
    image_bytes = client.generate_headshot(
        prompt="professional synthetic headshot\nsquare portrait"
    )

    assert image_bytes == SAMPLE_PNG_BYTES
    assert str(captured["url"]).startswith(
        "https://image.pollinations.ai/prompt/professional%20synthetic%20headshot%0Asquare%20portrait"
    )
    assert captured["params"] == {
        "width": 1024,
        "height": 1024,
        "model": "flux",
        "nologo": "true",
        "enhance": "true",
    }
    assert captured["timeout"] == 120


def test_huggingface_client_builds_json_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class FakeHeaders:
        def get(self, _key: str, default: str = "") -> str:
            _ = default
            return "image/png"

    class FakeResponse:
        headers = FakeHeaders()
        content = SAMPLE_PNG_BYTES

        def raise_for_status(self) -> None:
            return None

        def read(self) -> bytes:
            return SAMPLE_PNG_BYTES

        def __enter__(self) -> Self:
            return self

        def __exit__(
            self,
            exc_type: object,
            exc: object,
            tb: object,
        ) -> None:
            _ = (exc_type, exc, tb)

    def fake_post(
        url: str,
        *,
        json: dict[str, object],
        headers: dict[str, str],
        timeout: int,
    ) -> FakeResponse:
        captured["url"] = url
        captured["headers"] = headers
        captured["body"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        "cv_screener.cv_generation.photos.service.httpx.post",
        fake_post,
    )

    client = HuggingFaceHeadshotGenerationClient(
        GenerationConfig(huggingface_api_key="hf-secret")
    )
    image_bytes = client.generate_headshot(prompt="professional synthetic headshot")

    assert image_bytes == SAMPLE_PNG_BYTES
    assert str(captured["url"]).endswith("stabilityai/stable-diffusion-xl-base-1.0")
    assert captured["timeout"] == 120
    assert captured["body"] == {
        "inputs": "professional synthetic headshot",
        "parameters": {
            "width": 1024,
            "height": 1024,
        },
    }
