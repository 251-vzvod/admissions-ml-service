"""Output schema definitions for explainable scoring responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class EvidenceSpan(BaseModel):
    source: str
    snippet: str


class ExplanationNotes(BaseModel):
    potential: str
    motivation: str
    confidence: str
    authenticity_risk: str
    recommendation: str


class ExplanationPayload(BaseModel):
    summary: str
    scoring_notes: ExplanationNotes


class ScoreResponse(BaseModel):
    candidate_id: str
    scoring_run_id: str
    scoring_version: str
    prompt_version: str | None = None
    extraction_mode: str = "hybrid"
    extractor_version: str = "llm-extractor-v1"
    llm_metadata: dict[str, str | int | float] | None = None

    eligibility_status: str
    eligibility_reasons: list[str] = Field(default_factory=list)

    merit_score: int
    confidence_score: int
    authenticity_risk: int

    recommendation: str
    review_flags: list[str] = Field(default_factory=list)

    merit_breakdown: dict[str, int] = Field(default_factory=dict)
    feature_snapshot: dict[str, float | bool | int] = Field(default_factory=dict)

    top_strengths: list[str] = Field(default_factory=list)
    main_gaps: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    explanation: ExplanationPayload


class BatchScoreResponse(BaseModel):
    scoring_run_id: str
    scoring_version: str
    count: int
    results: list[ScoreResponse]
