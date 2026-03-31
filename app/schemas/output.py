"""Output schema definitions for explainable scoring responses."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.schemas.decision import Recommendation, ReviewFlag


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


class AIDetectorPayload(BaseModel):
    enabled: bool
    applicable: bool
    language: str | None = None
    probability_ai_generated: float | None = None
    provider: str
    model: str
    note: str | None = None


class ScoreResponse(BaseModel):
    candidate_id: str
    scoring_run_id: str
    scoring_version: str
    extraction_mode: str = "hybrid"
    llm_metadata: dict[str, str | int | float] | None = None

    eligibility_status: str
    eligibility_reasons: list[str] = Field(default_factory=list)

    merit_score: int
    confidence_score: int
    authenticity_risk: int

    recommendation: Recommendation
    review_flags: list[ReviewFlag] = Field(default_factory=list)

    merit_breakdown: dict[str, int] = Field(default_factory=dict)
    semantic_rubric_scores: dict[str, int] = Field(default_factory=dict)
    hidden_potential_score: int
    support_needed_score: int
    shortlist_priority_score: int
    evidence_coverage_score: int
    trajectory_score: int

    top_strengths: list[str] = Field(default_factory=list)
    main_gaps: list[str] = Field(default_factory=list)
    uncertainties: list[str] = Field(default_factory=list)
    authenticity_review_reasons: list[str] = Field(default_factory=list)
    ai_detector: AIDetectorPayload | None = None
    committee_cohorts: list[str] = Field(default_factory=list)
    why_candidate_surfaced: list[str] = Field(default_factory=list)
    what_to_verify_manually: list[str] = Field(default_factory=list)
    suggested_follow_up_question: str | None = None
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list)
    explanation: ExplanationPayload


class BatchScoreResponse(BaseModel):
    scoring_run_id: str
    scoring_version: str
    count: int
    ranked_candidate_ids: list[str] = Field(default_factory=list)
    shortlist_candidate_ids: list[str] = Field(default_factory=list)
    hidden_potential_candidate_ids: list[str] = Field(default_factory=list)
    support_needed_candidate_ids: list[str] = Field(default_factory=list)
    authenticity_review_candidate_ids: list[str] = Field(default_factory=list)
    results: list[ScoreResponse]
