"""Validation and parsing for LLM explainability outputs."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator


ALLOWED_SOURCES = {
    "motivation_letter_text",
    "motivation_questions",
    "interview_text",
    "video_interview_transcript_text",
    "video_presentation_transcript_text",
}

SOURCE_ALIASES = {
    "motivation_letter": "motivation_letter_text",
    "motivation": "motivation_letter_text",
    "letter": "motivation_letter_text",
    "motivation_questions_text": "motivation_questions",
    "questions": "motivation_questions",
    "qa": "motivation_questions",
    "interview": "interview_text",
    "video_interview": "video_interview_transcript_text",
    "video_presentation": "video_presentation_transcript_text",
    "presentation": "video_presentation_transcript_text",
}


def _normalize_source(raw: Any) -> str:
    value = "" if raw is None else str(raw).strip().lower()
    value = SOURCE_ALIASES.get(value, value)
    if value not in ALLOWED_SOURCES:
        return "motivation_letter_text"
    return value


class LLMClaimEvidence(BaseModel):
    claim: str
    source: str
    snippet: str

    @field_validator("claim", "snippet", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).strip()

    @field_validator("source", mode="before")
    @classmethod
    def normalize_source(cls, value: Any) -> str:
        return _normalize_source(value)


class LLMEvidenceSpan(BaseModel):
    dimension: str
    source: str
    text: str

    @model_validator(mode="before")
    @classmethod
    def normalize_span_payload(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        # Some models return snippet/quote instead of text.
        if "text" not in normalized:
            normalized["text"] = normalized.get("snippet") or normalized.get("quote") or ""
        if "dimension" not in normalized:
            normalized["dimension"] = normalized.get("type") or "explainability"
        return normalized

    @field_validator("dimension", "text", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).strip()

    @field_validator("source", mode="before")
    @classmethod
    def normalize_source(cls, value: Any) -> str:
        return _normalize_source(value)


class LLMExplainabilityOutput(BaseModel):
    top_strength_signals: list[LLMClaimEvidence] = Field(default_factory=list)
    main_gap_signals: list[LLMClaimEvidence] = Field(default_factory=list)
    uncertainties: list[LLMClaimEvidence] = Field(default_factory=list)
    evidence_spans: list[LLMEvidenceSpan] = Field(default_factory=list)
    extractor_rationale: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_root_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data

        normalized = dict(data)

        if "top_strength_signals" not in normalized:
            normalized["top_strength_signals"] = normalized.get("top_strengths") or normalized.get("strengths") or []
        if "main_gap_signals" not in normalized:
            normalized["main_gap_signals"] = normalized.get("main_gaps") or normalized.get("gaps") or []
        if "uncertainties" not in normalized:
            normalized["uncertainties"] = normalized.get("uncertainty_signals") or normalized.get("risks") or []
        if "evidence_spans" not in normalized:
            normalized["evidence_spans"] = normalized.get("evidence") or normalized.get("spans") or []
        if "extractor_rationale" not in normalized:
            normalized["extractor_rationale"] = normalized.get("rationale") or ""

        return normalized

    @field_validator("extractor_rationale", mode="before")
    @classmethod
    def normalize_rationale(cls, value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).strip()


class LLMParseError(RuntimeError):
    """Raised when LLM output cannot be parsed into schema."""


def _extract_first_json_object(raw_text: str) -> dict[str, Any] | None:
    """Extract first balanced JSON object from wrapper text responses."""
    start = raw_text.find("{")
    if start < 0:
        return None

    depth = 0
    in_string = False
    escaped = False

    for idx in range(start, len(raw_text)):
        ch = raw_text[idx]

        if in_string:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                snippet = raw_text[start : idx + 1]
                try:
                    parsed = json.loads(snippet)
                except json.JSONDecodeError:
                    return None
                return parsed if isinstance(parsed, dict) else None

    return None


def parse_llm_extraction_json(raw_text: str) -> LLMExplainabilityOutput:
    """Parse and validate LLM JSON output into explainability schema."""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        extracted = _extract_first_json_object(raw_text)
        if extracted is None:
            raise LLMParseError("invalid_json_response") from exc
        payload = extracted

    try:
        return LLMExplainabilityOutput.model_validate(payload)
    except ValidationError as exc:
        raise LLMParseError("invalid_explainability_schema") from exc
