"""Output schema definitions for explainable scoring responses."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from app.schemas.decision import Recommendation, ReviewFlag


class EvidenceSpan(BaseModel):
    source: str
    snippet: str


class ClaimEvidenceItem(BaseModel):
    claim: str
    support_level: str
    source: str
    snippet: str
    support_score: int
    rationale: str


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


class AIDetectorTextSourcePayload(BaseModel):
    source_key: str
    source_label: str
    applicable: bool
    probability_ai_generated: float | None = None
    note: str | None = None
    question: str | None = None


class CandidateTextAIProbabilitiesPayload(BaseModel):
    motivation_letter_text: AIDetectorTextSourcePayload | None = None
    interview_text: AIDetectorTextSourcePayload | None = None
    video_interview_transcript_text: AIDetectorTextSourcePayload | None = None
    video_presentation_transcript_text: AIDetectorTextSourcePayload | None = None
    motivation_questions: list[AIDetectorTextSourcePayload] = Field(default_factory=list)


class LLMRubricAssessmentPayload(BaseModel):
    leadership_potential: int | None = None
    growth_trajectory: int | None = None
    motivation_authenticity: int | None = None
    evidence_strength: int | None = None
    hidden_potential_hint: int | None = None
    authenticity_review_needed: str | None = None


class ReviewRoutingShadowPayload(BaseModel):
    enabled: bool
    available: bool
    artifact_name: str | None = None
    artifact_version: str | None = None
    target_name: str | None = None
    model_name: str | None = None
    probability: float | None = None
    threshold: float | None = None
    predicted_positive: bool | None = None
    note: str | None = None


class ScoreResponse(BaseModel):
    candidate_id: str
    scoring_run_id: str
    scoring_version: str

    eligibility_status: str
    eligibility_reasons: list[str] = Field(default_factory=list)

    merit_score: int
    confidence_score: int
    authenticity_risk: int
    ai_probability_ai_generated: float | None = None

    recommendation: Recommendation
    review_flags: list[ReviewFlag] = Field(default_factory=list)

    hidden_potential_score: int
    support_needed_score: int
    shortlist_priority_score: int
    evidence_coverage_score: int
    trajectory_score: int

    committee_cohorts: list[str] = Field(default_factory=list)
    why_candidate_surfaced: list[str] = Field(default_factory=list)
    what_to_verify_manually: list[str] = Field(default_factory=list)
    suggested_follow_up_question: str | None = None
    text_ai_probabilities: CandidateTextAIProbabilitiesPayload | None = None
    evidence_highlights: list[ClaimEvidenceItem] = Field(default_factory=list)
    top_strengths: list[str] = Field(default_factory=list)
    main_gaps: list[str] = Field(default_factory=list)
    explanation: ExplanationPayload

    extraction_mode: str = Field(default="hybrid", exclude=True)
    llm_metadata: dict[str, str | int | float] | None = Field(default=None, exclude=True)
    llm_rubric_assessment: LLMRubricAssessmentPayload | None = Field(default=None, exclude=True)
    merit_breakdown: dict[str, int] = Field(default_factory=dict, exclude=True)
    semantic_rubric_scores: dict[str, int] = Field(default_factory=dict, exclude=True)
    supported_claims: list[ClaimEvidenceItem] = Field(default_factory=list, exclude=True)
    weakly_supported_claims: list[ClaimEvidenceItem] = Field(default_factory=list, exclude=True)
    uncertainties: list[str] = Field(default_factory=list, exclude=True)
    authenticity_review_reasons: list[str] = Field(default_factory=list, exclude=True)
    ai_detector: AIDetectorPayload | None = Field(default=None, exclude=True)
    evidence_spans: list[EvidenceSpan] = Field(default_factory=list, exclude=True)
    review_routing_shadow: ReviewRoutingShadowPayload | None = Field(default=None, exclude=True)


class BatchScoreResponse(BaseModel):
    scoring_run_id: str
    scoring_version: str
    count: int
    results: list[ScoreResponse]


class RankedCandidateSummary(BaseModel):
    candidate_id: str
    rank_position: int
    recommendation: Recommendation
    merit_score: int
    confidence_score: int
    authenticity_risk: int
    ai_probability_ai_generated: float | None = None
    shortlist_priority_score: int
    hidden_potential_score: int
    support_needed_score: int
    evidence_coverage_score: int
    trajectory_score: int
    is_shortlist_candidate: bool = False
    is_hidden_potential_candidate: bool = False
    is_support_needed_candidate: bool = False
    is_authenticity_review_candidate: bool = False


class RankResponse(BaseModel):
    scoring_run_id: str
    scoring_version: str
    count: int
    returned_count: int
    ranked_candidate_ids: list[str] = Field(default_factory=list)
    ranked_candidates: list[RankedCandidateSummary] = Field(default_factory=list)
    shortlist_candidate_ids: list[str] = Field(default_factory=list)
    hidden_potential_candidate_ids: list[str] = Field(default_factory=list)
    support_needed_candidate_ids: list[str] = Field(default_factory=list)
    authenticity_review_candidate_ids: list[str] = Field(default_factory=list)
    ranker_metadata: dict[str, Any] = Field(default_factory=dict)
