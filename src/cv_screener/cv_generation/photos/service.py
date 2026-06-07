"""Generate synthetic CV headshots and persist them alongside YAML profiles."""

from __future__ import annotations

import base64
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import monotonic, sleep
from typing import TYPE_CHECKING, Protocol
from urllib.parse import quote

import httpx
from loguru import logger
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI, RateLimitError

from cv_screener.config import AppSettings, GenerationConfig, ImageGenerationProvider
from cv_screener.cv_generation.content.yaml_io import load_cv_profile, write_cv_profile

if TYPE_CHECKING:
    from cv_screener.cv_generation.content.schema import CVProfile


@dataclass(slots=True)
class PhotoGenerationSummary:
    """Counts for a headshot generation pass."""

    profile_count: int
    generated_count: int
    skipped_count: int


class HeadshotGenerationClient(Protocol):
    """Behavior needed from a headshot generation backend."""

    def generate_headshot(self, *, prompt: str) -> bytes:
        """Generate a PNG image for the supplied prompt."""


class OpenAIHeadshotGenerationClient:
    """OpenAI-compatible client for synthetic CV headshots."""

    def __init__(self, settings: GenerationConfig) -> None:
        """Initialize the image-generation client."""
        self.settings = settings
        self.client = OpenAI(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key.get_secret_value(),
        )

    def generate_headshot(self, *, prompt: str) -> bytes:
        """Generate a single headshot as PNG bytes."""
        response = self.client.images.generate(
            model=self.settings.image_generation_model,
            prompt=prompt,
            size=self.settings.image_generation_size,
            response_format="b64_json",
        )
        if not response.data:
            message = "image generation returned no image data"
            raise ValueError(message)
        data = response.data[0]
        image_b64 = getattr(data, "b64_json", None)
        if not image_b64:
            message = "image generation returned no base64 payload"
            raise ValueError(message)
        return base64.b64decode(image_b64)


class PollinationsHeadshotGenerationClient:
    """Pollinations client for synthetic CV headshots."""

    def __init__(self, settings: GenerationConfig) -> None:
        """Initialize the Pollinations client."""
        self.settings = settings

    def generate_headshot(self, *, prompt: str) -> bytes:
        """Generate a single headshot as PNG bytes."""
        width, height = _parse_image_size(self.settings.image_generation_size)
        prompt_segment = quote(prompt, safe="")
        try:
            response = httpx.get(
                f"{self.settings.pollinations_base_url}/{prompt_segment}",
                params={
                    "width": width,
                    "height": height,
                    "model": self.settings.pollinations_model,
                    "nologo": "true",
                    "enhance": "true",
                },
                headers={"Accept": "image/png"},
                timeout=120,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            message = f"pollinations image request failed: {error}"
            raise ValueError(message) from error
        return response.content


class HuggingFaceHeadshotGenerationClient:
    """Hugging Face inference client for synthetic CV headshots."""

    def __init__(self, settings: GenerationConfig) -> None:
        """Initialize the Hugging Face client."""
        self.settings = settings

    def generate_headshot(self, *, prompt: str) -> bytes:
        """Generate a single headshot as PNG bytes."""
        width, height = _parse_image_size(self.settings.image_generation_size)
        payload = {
            "inputs": prompt,
            "parameters": {
                "width": width,
                "height": height,
            },
        }
        headers = {"Content-Type": "application/json"}
        if self.settings.huggingface_api_key is not None:
            headers["Authorization"] = (
                f"Bearer {self.settings.huggingface_api_key.get_secret_value()}"
            )
        try:
            response = httpx.post(
                f"{self.settings.huggingface_base_url}/"
                f"{self.settings.huggingface_model}",
                json=payload,
                headers=headers,
                timeout=120,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            message = f"huggingface image request failed: {error}"
            raise ValueError(message) from error

        content_type = response.headers.get("content-type", "")
        if "json" in content_type:
            try:
                payload_data = response.json()
            except json.JSONDecodeError as error:
                message = "huggingface response was not valid JSON"
                raise ValueError(message) from error
            error_message = payload_data.get("error")
            if error_message:
                message = f"huggingface image request failed: {error_message}"
                raise ValueError(message)
            message = "huggingface returned JSON instead of image bytes"
            raise ValueError(message)
        return response.content


class CVPhotoGenerationService:
    """Generate and persist CV photos while updating the source YAML files."""

    def __init__(
        self,
        *,
        photo_dir: Path,
        client: HeadshotGenerationClient | None = None,
        settings: GenerationConfig | None = None,
    ) -> None:
        """Initialize the photo generation service."""
        self.photo_dir = photo_dir
        self.settings = settings or AppSettings().generation
        self.client = client or self._build_client(self.settings)
        self._request_lock = Lock()
        self._last_request_at = 0.0

    def generate_directory(self, directory: Path) -> PhotoGenerationSummary:
        """Generate photos for every YAML profile in a directory."""
        return self.generate_files(sorted(directory.glob("*.yaml")))

    def generate_files(self, paths: list[Path]) -> PhotoGenerationSummary:
        """Generate photos for specific YAML profiles."""
        if not paths:
            return PhotoGenerationSummary(
                profile_count=0,
                generated_count=0,
                skipped_count=0,
            )

        logger.info(
            "Generating CV photos",
            count=len(paths),
            photo_dir=str(self.photo_dir),
            client_type=type(self.client).__name__,
        )
        worker_count = max(
            1,
            min(len(paths), int(self.settings.image_generation_max_concurrency)),
        )
        generated_count = 0
        skipped_count = 0

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            future_to_path = {
                executor.submit(self._generate_one, path): path for path in paths
            }
            for future in as_completed(future_to_path):
                generated = future.result()
                if generated:
                    generated_count += 1
                else:
                    skipped_count += 1

        summary = PhotoGenerationSummary(
            profile_count=len(paths),
            generated_count=generated_count,
            skipped_count=skipped_count,
        )
        logger.info(
            "Completed CV photo generation",
            count=summary.profile_count,
            generated=summary.generated_count,
            skipped=summary.skipped_count,
            photo_dir=str(self.photo_dir),
        )
        return summary

    def _generate_one(self, path: Path) -> bool:
        profile = load_cv_profile(path)
        if self._has_valid_photo(profile):
            logger.debug(
                "Skipping CV photo generation",
                input_path=str(path),
                candidate_id=str(profile.candidate_id),
                photo_path=profile.photo_path,
            )
            return False

        photo_path = self._photo_path_for(profile)
        prompt = self._build_prompt(profile)
        logger.info(
            "Generating CV headshot",
            input_path=str(path),
            output_path=str(photo_path),
            candidate_id=str(profile.candidate_id),
        )
        image_bytes = self._generate_with_retries(prompt)
        photo_path.parent.mkdir(parents=True, exist_ok=True)
        photo_path.write_bytes(image_bytes)
        profile.photo_path = self._serialize_photo_path(photo_path)
        write_cv_profile(path, profile)
        return True

    def _generate_with_retries(self, prompt: str) -> bytes:
        last_error: str | None = None
        for attempt in range(self.settings.generation_max_retries + 1):
            try:
                self._wait_for_rate_limit_slot()
                return self.client.generate_headshot(prompt=prompt)
            except (
                APIConnectionError,
                APIError,
                APITimeoutError,
                RateLimitError,
                ValueError,
            ) as error:
                last_error = str(error)
                logger.warning(
                    "Failed to generate CV headshot",
                    attempt=attempt + 1,
                    max_attempts=self.settings.generation_max_retries + 1,
                    error_type=type(error).__name__,
                    error_message=last_error,
                )
                if attempt < self.settings.generation_max_retries:
                    sleep(self._retry_delay_seconds(attempt))
        message = (
            f"model failed to produce a valid CV headshot after retries: {last_error}"
        )
        raise ValueError(message)

    def _wait_for_rate_limit_slot(self) -> None:
        interval = self.settings.generation_min_interval_seconds
        if interval <= 0:
            return
        with self._request_lock:
            now = monotonic()
            elapsed = now - self._last_request_at
            if elapsed < interval:
                sleep(interval - elapsed)
            self._last_request_at = monotonic()

    def _retry_delay_seconds(self, attempt: int) -> float:
        base = self.settings.generation_retry_base_delay_seconds
        cap = self.settings.generation_retry_max_delay_seconds
        backoff = min(cap, base * (2**attempt))
        jitter = max(0.001, backoff * 0.1) * ((attempt % 3) + 1) / 3
        return backoff + jitter

    def _has_valid_photo(self, profile: CVProfile) -> bool:
        if not profile.photo_path:
            return False
        return Path(profile.photo_path).exists()

    def _photo_path_for(self, profile: CVProfile) -> Path:
        return self.photo_dir / f"{profile.candidate_id}.png"

    @staticmethod
    def _serialize_photo_path(photo_path: Path) -> str:
        try:
            return str(photo_path.relative_to(Path.cwd()))
        except ValueError:
            return str(photo_path)

    @staticmethod
    def _build_prompt(profile: CVProfile) -> str:
        return (
            "Create a professional synthetic headshot for a fictional CV.\n"
            f"Subject: {profile.full_name} from {profile.location}.\n"
            f"Background: neutral studio backdrop.\n"
            f"Career context: {profile.professional_summary}.\n"
            "Style: realistic shoulders-up portrait, clean business-casual attire, "
            "soft even lighting, natural expression, sharp focus, square composition.\n"
            "Constraints: fictional person only, no resemblance to any real person, "
            "celebrity, or copyrighted character; no text, no watermark, no logo, "
            "no props, no extra people."
        )

    @staticmethod
    def _build_client(settings: GenerationConfig) -> HeadshotGenerationClient:
        provider = settings.image_generation_provider
        if provider is ImageGenerationProvider.POLLINATIONS:
            return PollinationsHeadshotGenerationClient(settings)
        if provider is ImageGenerationProvider.HUGGINGFACE:
            return HuggingFaceHeadshotGenerationClient(settings)
        return OpenAIHeadshotGenerationClient(settings)


def _parse_image_size(size: str) -> tuple[int, int]:
    width_text, height_text = size.lower().split("x", maxsplit=1)
    return int(width_text), int(height_text)
