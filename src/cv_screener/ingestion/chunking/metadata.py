"""Metadata extraction for chunk text."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, cast

import phonenumbers
from email_validator import EmailNotValidError, validate_email
from gliner import GLiNER
from phonenumbers import PhoneNumberFormat

from cv_screener.ingestion.chunking.common import (
    _COMPANY_LABEL,
    _GITHUB_PATTERN,
    _LINKEDIN_PATTERN,
    _PERSON_LABEL,
    _SKILL_LABEL,
    _UNIVERSITY_LABEL,
    dedupe,
)
from cv_screener.ingestion.schema import ChunkConfig, ParsedCV

if TYPE_CHECKING:
    from phonenumbers.phonenumbermatcher import PhoneNumberMatch


class _EntityPredictor(Protocol):
    def predict_entities(
        self,
        text: str,
        labels: list[str],
        *,
        threshold: float,
    ) -> list[dict[str, str | float]]: ...

@dataclass(frozen=True)
class ChunkExtractionResult:
    """Extracted entities and contact metadata for one chunk."""

    detected_skills: list[str]
    detected_companies: list[str]
    detected_universities: list[str]
    email_addresses: list[str]
    phone_numbers: list[str]
    linkedin_urls: list[str]
    github_urls: list[str]


class MetadataExtractor:
    """Extract chunk metadata with GLiNER plus deterministic contact patterns."""

    def __init__(self, config: ChunkConfig) -> None:
        """Configure the extractor with the selected GLiNER model and thresholds."""
        self._config = config
        self._model: _EntityPredictor | None = None

    def _get_model(self) -> _EntityPredictor:
        if self._model is None:
            self._model = cast(
                "_EntityPredictor",
                GLiNER.from_pretrained(self._config.gliner_model_name),
            )
        return self._model

    def _predict_entities(
        self,
        text: str,
        labels: list[str],
    ) -> list[dict[str, str | float]]:
        if not text.strip():
            return []
        model = self._get_model()
        return model.predict_entities(
            text,
            labels,
            threshold=self._config.gliner_threshold,
        )

    def candidate_name(self, cv: ParsedCV) -> str | None:
        """Extract the best candidate-name match from the top of the first page."""
        first_page_text = cv.pages[0].markdown[:2000]
        entities = self._predict_entities(first_page_text, [_PERSON_LABEL])
        if not entities:
            return None
        best_match = max(
            entities,
            key=lambda entity: float(entity.get("score", 0.0)),
        )
        text = str(best_match["text"]).strip()
        return text or None

    def extract_chunk_metadata(self, text: str) -> ChunkExtractionResult:
        """Extract all chunk-level metadata from text."""
        entities = self._predict_entities(
            text,
            [_SKILL_LABEL, _COMPANY_LABEL, _UNIVERSITY_LABEL],
        )
        skills = dedupe(
            str(entity["text"])
            for entity in entities
            if str(entity["label"]).casefold() == _SKILL_LABEL
        )
        companies = dedupe(
            str(entity["text"])
            for entity in entities
            if str(entity["label"]).casefold() == _COMPANY_LABEL
        )
        universities = dedupe(
            str(entity["text"])
            for entity in entities
            if str(entity["label"]).casefold() == _UNIVERSITY_LABEL
        )
        return ChunkExtractionResult(
            detected_skills=skills,
            detected_companies=companies,
            detected_universities=universities,
            email_addresses=self._extract_email_addresses(text),
            phone_numbers=self._extract_phone_numbers(text),
            linkedin_urls=dedupe(
                match.group(0) for match in _LINKEDIN_PATTERN.finditer(text)
            ),
            github_urls=dedupe(
                match.group(0) for match in _GITHUB_PATTERN.finditer(text)
            ),
        )

    def _extract_email_addresses(self, text: str) -> list[str]:
        emails: list[str] = []
        for token in text.split():
            candidate = token.strip("()[]{}<>,;:\"'")
            if "@" not in candidate:
                continue
            try:
                normalized = validate_email(
                    candidate,
                    check_deliverability=False,
                ).normalized
            except EmailNotValidError:
                continue
            emails.append(normalized)
        return dedupe(emails)

    def _extract_phone_numbers(self, text: str) -> list[str]:
        matches = cast(
            "list[PhoneNumberMatch]",
            list(phonenumbers.PhoneNumberMatcher(text, None)),
        )
        valid_numbers = [
            phonenumbers.format_number(match.number, PhoneNumberFormat.E164)
            for match in matches
            if phonenumbers.is_valid_number(match.number)
        ]
        return dedupe(valid_numbers)
