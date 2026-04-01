"""Input schema definitions for candidate scoring endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _build_profile_dict(values: dict[str, Any]) -> dict[str, Any]:
    profile_raw = _as_dict(values.get("profile"))
    legacy_structured = _as_dict(values.get("structured_data"))
    nested_structured = _as_dict(profile_raw.get("structured_data"))

    academics = (
        profile_raw.get("academics")
        if isinstance(profile_raw.get("academics"), dict)
        else nested_structured.get("education")
        if isinstance(nested_structured.get("education"), dict)
        else legacy_structured.get("education")
        if isinstance(legacy_structured.get("education"), dict)
        else None
    )
    materials = (
        profile_raw.get("materials")
        if isinstance(profile_raw.get("materials"), dict)
        else nested_structured.get("application_materials")
        if isinstance(nested_structured.get("application_materials"), dict)
        else legacy_structured.get("application_materials")
        if isinstance(legacy_structured.get("application_materials"), dict)
        else None
    )
    narratives = (
        profile_raw.get("narratives")
        if isinstance(profile_raw.get("narratives"), dict)
        else profile_raw.get("application_text")
        if isinstance(profile_raw.get("application_text"), dict)
        else profile_raw.get("text_inputs")
        if isinstance(profile_raw.get("text_inputs"), dict)
        else values.get("text_inputs")
        if isinstance(values.get("text_inputs"), dict)
        else {}
    )
    process_signals = (
        profile_raw.get("process_signals")
        if isinstance(profile_raw.get("process_signals"), dict)
        else profile_raw.get("behavioral_signals")
        if isinstance(profile_raw.get("behavioral_signals"), dict)
        else values.get("behavioral_signals")
        if isinstance(values.get("behavioral_signals"), dict)
        else None
    )
    metadata = (
        profile_raw.get("metadata")
        if isinstance(profile_raw.get("metadata"), dict)
        else values.get("metadata")
        if isinstance(values.get("metadata"), dict)
        else None
    )

    return {
        "academics": academics,
        "materials": materials,
        "narratives": narratives,
        "process_signals": process_signals,
        "metadata": metadata,
    }


def normalize_candidate_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize canonical and legacy payloads into one internal runtime shape."""
    values = dict(payload) if isinstance(payload, dict) else {}
    profile = _build_profile_dict(values)
    candidate_id = values.get("candidate_id")
    if isinstance(candidate_id, str):
        candidate_id = candidate_id.strip()

    return {
        "candidate_id": candidate_id or "",
        "consent": values.get("consent"),
        "metadata": profile.get("metadata"),
        "profile": profile,
        "structured_data": {
            "education": profile.get("academics"),
            "application_materials": profile.get("materials"),
        },
        "text_inputs": profile.get("narratives") or {},
        "behavioral_signals": profile.get("process_signals") or {},
    }


class EnglishProficiency(BaseModel):
    type: str | None = Field(default=None)
    score: float | None = Field(default=None)

    model_config = {"extra": "forbid"}


class SchoolCertificate(BaseModel):
    type: str | None = Field(default=None)
    score: float | None = Field(default=None)

    model_config = {"extra": "forbid"}


class AcademicsInput(BaseModel):
    english_proficiency: EnglishProficiency | None = None
    school_certificate: SchoolCertificate | None = None

    model_config = {"extra": "forbid"}


class ApplicationMaterialsInput(BaseModel):
    documents: list[str] = Field(default_factory=list)
    attachments: list[str] = Field(default_factory=list)
    portfolio_links: list[str] = Field(default_factory=list)
    video_presentation_link: str | None = None
    videoPresentationLink: str | None = None
    video_url: str | None = None

    model_config = {"extra": "forbid"}


class MotivationAnswer(BaseModel):
    question: str | None = None
    answer: str | None = None

    model_config = {"extra": "forbid"}


class NarrativeInputs(BaseModel):
    motivation_letter_text: str | None = None
    motivation_questions: list[MotivationAnswer] = Field(default_factory=list)
    interview_text: str | None = None
    video_interview_transcript_text: str | None = None
    video_presentation_transcript_text: str | None = None

    model_config = {"extra": "forbid"}


class ProcessSignals(BaseModel):
    completion_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    returned_to_edit: bool | None = None
    skipped_optional_questions: int | None = Field(default=None, ge=0)
    meaningful_answers_count: int | None = Field(default=None, ge=0)
    scenario_depth: float | None = Field(default=None, ge=0.0, le=1.0)

    model_config = {"extra": "forbid"}


class CandidateMetadata(BaseModel):
    source: str | None = None
    submitted_at: str | None = None
    scoring_version: str | None = None

    model_config = {"extra": "forbid"}


class CandidateProfile(BaseModel):
    academics: AcademicsInput | None = None
    materials: ApplicationMaterialsInput | None = None
    narratives: NarrativeInputs = Field(default_factory=NarrativeInputs)
    process_signals: ProcessSignals | None = None
    metadata: CandidateMetadata | None = None

    model_config = {"extra": "forbid"}


class StructuredDataInput(BaseModel):
    education: AcademicsInput | None = None
    application_materials: ApplicationMaterialsInput | None = None

    model_config = {"extra": "forbid"}


class CandidateInput(BaseModel):
    candidate_id: str = Field(min_length=1)
    structured_data: StructuredDataInput = Field(default_factory=StructuredDataInput)
    text_inputs: NarrativeInputs = Field(default_factory=NarrativeInputs)
    behavioral_signals: ProcessSignals | None = None
    metadata: CandidateMetadata | None = None
    consent: bool | None = None

    model_config = {"extra": "forbid"}

    @model_validator(mode="before")
    @classmethod
    def normalize_public_payload(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values

        normalized = dict(values)
        candidate_id = normalized.get("candidate_id")
        if isinstance(candidate_id, str):
            normalized["candidate_id"] = candidate_id.strip()

        profile = _build_profile_dict(normalized)
        normalized["structured_data"] = {
            "education": profile.get("academics"),
            "application_materials": profile.get("materials"),
        }
        normalized["text_inputs"] = profile.get("narratives") or {}
        normalized["behavioral_signals"] = profile.get("process_signals")
        normalized["metadata"] = profile.get("metadata")
        normalized.pop("profile", None)
        return normalized


class BatchScoreRequest(BaseModel):
    candidates: list[CandidateInput]

    model_config = {"extra": "forbid"}


class ScoreFileRequest(BaseModel):
    file_path: str

    model_config = {"extra": "forbid"}
