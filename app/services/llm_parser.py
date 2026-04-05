"""Validation and parsing for LLM calibration and explainability outputs."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from app.schemas.decision import Recommendation


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

ALLOWED_RECOMMENDATIONS = {item.value for item in Recommendation}


def _normalize_source(raw: Any) -> str:
    value = "" if raw is None else str(raw).strip().lower()
    value = SOURCE_ALIASES.get(value, value)
    if value not in ALLOWED_SOURCES:
        return "motivation_letter_text"
    return value


def _normalize_text(raw: Any) -> str:
    if raw is None:
        return ""
    return " ".join(str(raw).split()).strip()


def _normalize_question_text(raw: Any) -> str:
    if isinstance(raw, list):
        for item in raw:
            text = _normalize_question_text(item)
            if text:
                return text
        return ""
    if isinstance(raw, dict):
        for key in ("question", "text", "value", "content", "message", "summary"):
            if key in raw:
                text = _normalize_question_text(raw.get(key))
                if text:
                    return text
        return ""
    return _normalize_text(raw)


def _normalize_bool(raw: Any) -> bool:
    if isinstance(raw, bool):
        return raw
    value = _normalize_text(raw).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n", ""}:
        return False
    if value in {"medium", "high"}:
        return True
    if value == "low":
        return False
    return False


def _normalize_bounded_score(raw: Any) -> int | None:
    text_value = _normalize_text(raw).lower()
    textual_map = {
        "very weak": 1,
        "weak": 2,
        "mixed": 3,
        "moderate": 3,
        "medium": 3,
        "strong": 4,
        "high": 4,
        "very strong": 5,
    }
    if text_value in textual_map:
        return textual_map[text_value]
    try:
        value_int = int(raw)
    except (TypeError, ValueError):
        return None
    return max(1, min(5, value_int))


class LLMClaimEvidence(BaseModel):
    claim: str
    source: str
    snippet: str

    @field_validator("claim", "snippet", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        return _normalize_text(value)

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
        if "text" not in normalized:
            normalized["text"] = normalized.get("snippet") or normalized.get("quote") or ""
        if "dimension" not in normalized:
            normalized["dimension"] = normalized.get("type") or "explainability"
        return normalized

    @field_validator("dimension", "text", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        return _normalize_text(value)

    @field_validator("source", mode="before")
    @classmethod
    def normalize_source(cls, value: Any) -> str:
        return _normalize_source(value)


class LLMHumanReview(BaseModel):
    recommendation: str = ""
    shortlist_band: bool = False
    hidden_potential_band: bool = False
    support_needed_band: bool = False
    authenticity_review_band: bool = False
    notes: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_root(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "support_needed_band" not in normalized:
            normalized["support_needed_band"] = normalized.get("needs_support_band", normalized.get("needs_support_flag"))
        if "authenticity_review_band" not in normalized:
            normalized["authenticity_review_band"] = normalized.get(
                "authenticity_review_needed",
                normalized.get("authenticity_review"),
            )
        return normalized

    @field_validator("recommendation", mode="before")
    @classmethod
    def normalize_recommendation(cls, value: Any) -> str:
        lowered = _normalize_text(value).lower()
        return lowered if lowered in ALLOWED_RECOMMENDATIONS else ""

    @field_validator(
        "shortlist_band",
        "hidden_potential_band",
        "support_needed_band",
        "authenticity_review_band",
        mode="before",
    )
    @classmethod
    def normalize_bool(cls, value: Any) -> bool:
        return _normalize_bool(value)

    @field_validator("notes", mode="before")
    @classmethod
    def normalize_notes(cls, value: Any) -> str:
        return _normalize_text(value)


class LLMExplainabilityOutput(BaseModel):
    candidate_id: str = ""
    human_review: LLMHumanReview | None = None
    rubric: dict[str, Any] = Field(default_factory=dict)
    evidence_bullets: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)

    top_strength_signals: list[LLMClaimEvidence] = Field(default_factory=list)
    main_gap_signals: list[LLMClaimEvidence] = Field(default_factory=list)
    uncertainty_signals: list[LLMClaimEvidence] = Field(default_factory=list)
    evidence_spans: list[LLMEvidenceSpan] = Field(default_factory=list)
    extractor_rationale: str = ""
    rubric_assessment: dict[str, Any] = Field(default_factory=dict)
    committee_follow_up_question: str = ""

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
        if "evidence_spans" not in normalized:
            normalized["evidence_spans"] = normalized.get("evidence") or normalized.get("spans") or []
        if "extractor_rationale" not in normalized:
            normalized["extractor_rationale"] = (
                normalized.get("rationale")
                or normalized.get("answer")
                or normalized.get("response")
                or normalized.get("output")
                or ""
            )
        if "rubric_assessment" not in normalized:
            normalized["rubric_assessment"] = normalized.get("llm_rubric_assessment") or normalized.get("rubric") or {}
        if "committee_follow_up_question" not in normalized:
            normalized["committee_follow_up_question"] = normalized.get("follow_up_question") or ""
        if "human_review" not in normalized and isinstance(normalized.get("review"), dict):
            normalized["human_review"] = normalized["review"]
        if "top_strength_signals" not in normalized and isinstance(normalized.get("strength_signals"), list):
            normalized["top_strength_signals"] = normalized.get("strength_signals") or []
        if "main_gap_signals" not in normalized and isinstance(normalized.get("gap_signals"), list):
            normalized["main_gap_signals"] = normalized.get("gap_signals") or []
        if "uncertainty_signals" not in normalized and isinstance(normalized.get("uncertainty_items"), list):
            normalized["uncertainty_signals"] = normalized.get("uncertainty_items") or []

        raw_uncertainties = normalized.get("uncertainties")
        if isinstance(raw_uncertainties, list) and raw_uncertainties and all(isinstance(item, dict) for item in raw_uncertainties):
            normalized["uncertainty_signals"] = raw_uncertainties
            normalized["uncertainties"] = []
        elif "uncertainty_signals" not in normalized:
            normalized["uncertainty_signals"] = normalized.get("uncertainty_signals") or normalized.get("uncertainty_claims") or normalized.get("risks") or []
            if raw_uncertainties is None:
                normalized["uncertainties"] = []

        return normalized

    @field_validator("candidate_id", mode="before")
    @classmethod
    def normalize_candidate_id(cls, value: Any) -> str:
        return _normalize_text(value)

    @field_validator("evidence_bullets", "uncertainties", mode="before")
    @classmethod
    def normalize_string_list(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        for item in value:
            text = _normalize_text(item)
            if text:
                normalized.append(text)
        return normalized

    @field_validator("extractor_rationale", mode="before")
    @classmethod
    def normalize_rationale(cls, value: Any) -> str:
        return _normalize_text(value)

    @field_validator("committee_follow_up_question", mode="before")
    @classmethod
    def normalize_follow_up(cls, value: Any) -> str:
        return _normalize_question_text(value)

    @field_validator("rubric", mode="before")
    @classmethod
    def normalize_calibration_rubric(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}

        normalized = dict(value)
        aliases = {
            "leadership": "leadership_through_action",
            "leadership_potential": "leadership_through_action",
            "readiness": "project_based_readiness",
            "motivation": "motivation_groundedness",
            "motivation_authenticity": "motivation_groundedness",
        }
        for old_key, new_key in aliases.items():
            if old_key in normalized and new_key not in normalized:
                normalized[new_key] = normalized[old_key]

        bounded_scores = [
            "leadership_through_action",
            "growth_trajectory",
            "community_orientation",
            "project_based_readiness",
            "motivation_groundedness",
            "evidence_strength",
            "shortlist_priority",
        ]
        output: dict[str, Any] = {}
        for key in bounded_scores:
            normalized_value = _normalize_bounded_score(normalized.get(key))
            if normalized_value is not None:
                output[key] = normalized_value
        return output

    @field_validator("rubric_assessment", mode="before")
    @classmethod
    def normalize_rubric_assessment(cls, value: Any) -> dict[str, Any]:
        if not isinstance(value, dict):
            return {}

        normalized = dict(value)
        aliases = {
            "leadership": "leadership_potential",
            "growth": "growth_trajectory",
            "motivation": "motivation_authenticity",
            "evidence": "evidence_strength",
            "hidden_potential": "hidden_potential_hint",
            "authenticity_review": "authenticity_review_needed",
        }
        for old_key, new_key in aliases.items():
            if old_key in normalized and new_key not in normalized:
                normalized[new_key] = normalized[old_key]

        bounded_scores = [
            "leadership_potential",
            "growth_trajectory",
            "motivation_authenticity",
            "evidence_strength",
            "hidden_potential_hint",
        ]
        output: dict[str, Any] = {}
        for key in bounded_scores:
            normalized_value = _normalize_bounded_score(normalized.get(key))
            if normalized_value is not None:
                output[key] = normalized_value

        review_signal = normalized.get("authenticity_review_needed")
        lowered = _normalize_text(review_signal).lower()
        if lowered in {"low", "medium", "high"}:
            output["authenticity_review_needed"] = lowered
        return output


class LLMParseError(RuntimeError):
    """Raised when LLM output cannot be parsed into schema."""


class LLMCommitteeNarrativeOutput(BaseModel):
    summary: str = ""
    top_strengths: list[str] = Field(default_factory=list)
    main_gaps: list[str] = Field(default_factory=list)
    what_to_verify_manually: list[str] = Field(default_factory=list)
    suggested_follow_up_question: str = ""

    @model_validator(mode="before")
    @classmethod
    def normalize_root_keys(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        if "summary" not in normalized:
            normalized["summary"] = (
                normalized.get("brief_summary")
                or normalized.get("detailed_summary")
                or normalized.get("committee_rationale")
                or ""
            )
        if "top_strengths" not in normalized:
            normalized["top_strengths"] = normalized.get("strengths") or []
        if "main_gaps" not in normalized:
            normalized["main_gaps"] = normalized.get("gaps") or []
        if "what_to_verify_manually" not in normalized:
            normalized["what_to_verify_manually"] = (
                normalized.get("manual_verification_focus")
                or normalized.get("verify_manually")
                or []
            )
        if "suggested_follow_up_question" not in normalized:
            normalized["suggested_follow_up_question"] = normalized.get("follow_up_question") or ""
        return normalized

    @field_validator("summary", mode="before")
    @classmethod
    def normalize_summary(cls, value: Any) -> str:
        return _normalize_text(value)

    @field_validator("suggested_follow_up_question", mode="before")
    @classmethod
    def normalize_follow_up_question(cls, value: Any) -> str:
        return _normalize_question_text(value)

    @field_validator("top_strengths", "main_gaps", "what_to_verify_manually", mode="before")
    @classmethod
    def normalize_string_list(cls, value: Any) -> list[str]:
        if not isinstance(value, list):
            return []
        normalized: list[str] = []
        for item in value:
            text = _normalize_text(item)
            if text:
                normalized.append(text)
        return normalized[:3]


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
    """Parse and validate LLM JSON output into calibration/explainability schema."""
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


def parse_llm_committee_json(raw_text: str) -> LLMCommitteeNarrativeOutput:
    """Parse and validate LLM committee narrative JSON."""
    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        extracted = _extract_first_json_object(raw_text)
        if extracted is None:
            raise LLMParseError("invalid_json_response") from exc
        payload = extracted

    try:
        return LLMCommitteeNarrativeOutput.model_validate(payload)
    except ValidationError as exc:
        raise LLMParseError("invalid_committee_narrative_schema") from exc
