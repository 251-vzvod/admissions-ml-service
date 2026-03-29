"""Validation and parsing for LLM explainability outputs."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator


class LLMClaimEvidence(BaseModel):
    claim: str
    source: Literal["motivation_letter_text", "motivation_questions", "interview_text"]
    snippet: str

    @field_validator("claim", "snippet", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).strip()


class LLMEvidenceSpan(BaseModel):
    dimension: str
    source: Literal["motivation_letter_text", "motivation_questions", "interview_text"]
    text: str

    @field_validator("dimension", "text", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        if value is None:
            return ""
        return " ".join(str(value).split()).strip()


class LLMExplainabilityOutput(BaseModel):
    top_strength_signals: list[LLMClaimEvidence] = Field(default_factory=list)
    main_gap_signals: list[LLMClaimEvidence] = Field(default_factory=list)
    uncertainties: list[LLMClaimEvidence] = Field(default_factory=list)
    evidence_spans: list[LLMEvidenceSpan] = Field(default_factory=list)
    extractor_rationale: str = ""

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
