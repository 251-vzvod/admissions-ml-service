"""Validation and parsing for structured LLM extractor outputs."""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.utils.math_utils import clamp01


class LLMEvidenceSpan(BaseModel):
    dimension: str
    source: Literal["motivation_letter_text", "motivation_questions", "interview_text"]
    text: str


class LLMExtractionOutput(BaseModel):
    motivation_clarity: float
    initiative: float
    leadership_impact: float
    growth_trajectory: float
    resilience: float
    program_fit: float
    evidence_richness: float

    specificity_score: float
    evidence_count: float
    consistency_score: float
    completeness_score: float

    genericness_score: float
    contradiction_flag: bool
    polished_but_empty_score: float
    cross_section_mismatch_score: float

    top_strength_signals: list[str] = Field(default_factory=list)
    main_gap_signals: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    evidence_spans: list[LLMEvidenceSpan] = Field(default_factory=list)
    extractor_rationale: str = ""

    @field_validator(
        "motivation_clarity",
        "initiative",
        "leadership_impact",
        "growth_trajectory",
        "resilience",
        "program_fit",
        "evidence_richness",
        "specificity_score",
        "evidence_count",
        "consistency_score",
        "completeness_score",
        "genericness_score",
        "polished_but_empty_score",
        "cross_section_mismatch_score",
        mode="before",
    )
    @classmethod
    def clamp_numeric(cls, value: Any) -> float:
        try:
            return clamp01(float(value))
        except (TypeError, ValueError):
            return 0.0


class LLMParseError(RuntimeError):
    """Raised when LLM output cannot be parsed into schema."""


def parse_llm_extraction_json(raw_text: str) -> LLMExtractionOutput:
    """Parse and validate LLM JSON output into strict extraction schema."""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise LLMParseError("invalid_json_response") from exc

    try:
        return LLMExtractionOutput.model_validate(payload)
    except ValidationError as exc:
        raise LLMParseError("invalid_extraction_schema") from exc
