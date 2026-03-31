"""HTTP routes for scoring service."""

from __future__ import annotations

from fastapi import APIRouter

from app.config import CONFIG
from app.schemas.input import BatchScoreRequest, CandidateInput
from app.schemas.output import BatchScoreResponse, ScoreResponse
from app.services.pipeline import ScoringPipeline
from app.services.shortlist import build_batch_shortlist_summary
from app.utils.ids import generate_scoring_run_id


router = APIRouter()
pipeline = ScoringPipeline()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "invision-u-scoring-mvp", "scoring_version": CONFIG.scoring_version}


@router.post("/score", response_model=ScoreResponse)
def score_candidate(candidate: CandidateInput) -> ScoreResponse:
    return pipeline.score_candidate_model(candidate, enable_llm_explainability=CONFIG.llm.enabled)


@router.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(payload: BatchScoreRequest) -> BatchScoreResponse:
    scoring_run_id = generate_scoring_run_id()
    results = [
        pipeline.score_candidate_model(
            candidate,
            scoring_run_id=scoring_run_id,
            enable_llm_explainability=CONFIG.llm.enabled,
        )
        for candidate in payload.candidates
    ]
    shortlist_summary = build_batch_shortlist_summary(results)
    return BatchScoreResponse(
        scoring_run_id=scoring_run_id,
        scoring_version=CONFIG.scoring_version,
        count=len(results),
        ranked_candidate_ids=shortlist_summary.ranked_candidate_ids,
        shortlist_candidate_ids=shortlist_summary.shortlist_candidate_ids,
        hidden_potential_candidate_ids=shortlist_summary.hidden_potential_candidate_ids,
        support_needed_candidate_ids=shortlist_summary.support_needed_candidate_ids,
        authenticity_review_candidate_ids=shortlist_summary.authenticity_review_candidate_ids,
        results=results,
    )
