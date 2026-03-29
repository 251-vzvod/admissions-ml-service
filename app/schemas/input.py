"""Input schema definitions for candidate scoring endpoints."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class EnglishProficiency(BaseModel):
    type: str | None = Field(default=None)
    score: float | None = Field(default=None)


class SchoolCertificate(BaseModel):
    type: str | None = Field(default=None)
    score: float | None = Field(default=None)


class EducationInput(BaseModel):
    english_proficiency: EnglishProficiency | None = None
    school_certificate: SchoolCertificate | None = None


class StructuredDataInput(BaseModel):
    education: EducationInput | None = None


class MotivationAnswer(BaseModel):
    question: str | None = None
    answer: str | None = None


class TextInputs(BaseModel):
    motivation_letter_text: str | None = None
    motivation_questions: list[MotivationAnswer] = Field(default_factory=list)
    interview_text: str | None = None
    video_interview_transcript_text: str | None = None
    video_presentation_transcript_text: str | None = None


class BehavioralSignals(BaseModel):
    completion_rate: float | None = Field(default=None, ge=0.0, le=1.0)
    returned_to_edit: bool | None = None
    skipped_optional_questions: int | None = Field(default=None, ge=0)
    meaningful_answers_count: int | None = Field(default=None, ge=0)
    scenario_depth: float | None = Field(default=None, ge=0.0, le=1.0)


class CandidateMetadata(BaseModel):
    source: str | None = None
    submitted_at: str | None = None
    scoring_version: str | None = None


class CandidateInput(BaseModel):
    candidate_id: str = Field(min_length=1)
    structured_data: StructuredDataInput | None = None
    text_inputs: TextInputs = Field(default_factory=TextInputs)
    behavioral_signals: BehavioralSignals | None = None
    metadata: CandidateMetadata | None = None
    consent: bool | None = None

    model_config = {"extra": "allow"}

    @model_validator(mode="before")
    @classmethod
    def normalize_candidate_id(cls, values: Any) -> Any:
        if isinstance(values, dict) and "candidate_id" in values and isinstance(values["candidate_id"], str):
            values["candidate_id"] = values["candidate_id"].strip()
        return values


class BatchScoreRequest(BaseModel):
    candidates: list[CandidateInput]


class ScoreFileRequest(BaseModel):
    file_path: str
