"""HTTP routes for scoring service."""

from __future__ import annotations

from fastapi import APIRouter, Query

from app.config import CONFIG
from app.schemas.input import BatchScoreRequest, CandidateInput
from app.schemas.output import BatchScoreResponse, RankResponse, RankedCandidateSummary, ScoreResponse
from app.services.offline_ranker import get_offline_ranker_metadata, rank_results_with_offline_ranker
from app.services.pipeline import ScoringPipeline
from app.services.shortlist import build_batch_shortlist_summary
from app.utils.ids import generate_scoring_run_id


router = APIRouter()
pipeline = ScoringPipeline()


def _score_batch_candidates(payload: BatchScoreRequest) -> tuple[str, list[ScoreResponse]]:
    scoring_run_id = generate_scoring_run_id()
    results = [
        pipeline.score_candidate_model(
            candidate,
            scoring_run_id=scoring_run_id,
            enable_llm_explainability=CONFIG.llm.enabled,
        )
        for candidate in payload.candidates
    ]
    return scoring_run_id, results


def _build_ranked_candidate_summaries(
    results: list[ScoreResponse],
    *,
    top_k: int | None = None,
) -> tuple[list[RankedCandidateSummary], list[str]]:
    shortlist_summary = build_batch_shortlist_summary(results)
    sorted_results = rank_results_with_offline_ranker(results)
    if top_k is None:
        selected_results = sorted_results
    else:
        selected_results = sorted_results[: min(top_k, len(sorted_results))]

    shortlist_ids = set(shortlist_summary.shortlist_candidate_ids)
    hidden_ids = set(shortlist_summary.hidden_potential_candidate_ids)
    support_ids = set(shortlist_summary.support_needed_candidate_ids)
    authenticity_ids = set(shortlist_summary.authenticity_review_candidate_ids)

    ranked_candidates = [
        RankedCandidateSummary(
            candidate_id=item.candidate_id,
            rank_position=index,
            recommendation=item.recommendation,
            merit_score=item.merit_score,
            confidence_score=item.confidence_score,
            authenticity_risk=item.authenticity_risk,
            ai_probability_ai_generated=item.ai_probability_ai_generated,
            shortlist_priority_score=item.shortlist_priority_score,
            hidden_potential_score=item.hidden_potential_score,
            support_needed_score=item.support_needed_score,
            evidence_coverage_score=item.evidence_coverage_score,
            trajectory_score=item.trajectory_score,
            is_shortlist_candidate=item.candidate_id in shortlist_ids,
            is_hidden_potential_candidate=item.candidate_id in hidden_ids,
            is_support_needed_candidate=item.candidate_id in support_ids,
            is_authenticity_review_candidate=item.candidate_id in authenticity_ids,
        )
        for index, item in enumerate(selected_results, start=1)
    ]
    return ranked_candidates, shortlist_summary.ranked_candidate_ids


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "invision-u-scoring-mvp", "scoring_version": CONFIG.scoring_version}


@router.post("/score", response_model=ScoreResponse)
def score_candidate(candidate: CandidateInput) -> ScoreResponse:
    return pipeline.score_candidate_model(candidate, enable_llm_explainability=CONFIG.llm.enabled)


@router.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(payload: BatchScoreRequest) -> BatchScoreResponse:
    scoring_run_id, results = _score_batch_candidates(payload)
    return BatchScoreResponse(
        scoring_run_id=scoring_run_id,
        scoring_version=CONFIG.scoring_version,
        count=len(results),
        results=results,
    )


@router.post("/rank", response_model=RankResponse)
def rank_batch(payload: BatchScoreRequest, top_k: int | None = Query(default=None, ge=1)) -> RankResponse:
    scoring_run_id, results = _score_batch_candidates(payload)
    shortlist_summary = build_batch_shortlist_summary(results)
    ranked_candidates, full_ranked_candidate_ids = _build_ranked_candidate_summaries(results, top_k=top_k)
    returned_ranked_ids = [candidate.candidate_id for candidate in ranked_candidates]
    return RankResponse(
        scoring_run_id=scoring_run_id,
        scoring_version=CONFIG.scoring_version,
        count=len(results),
        returned_count=len(ranked_candidates),
        ranked_candidate_ids=returned_ranked_ids,
        ranked_candidates=ranked_candidates,
        shortlist_candidate_ids=shortlist_summary.shortlist_candidate_ids,
        hidden_potential_candidate_ids=shortlist_summary.hidden_potential_candidate_ids,
        support_needed_candidate_ids=shortlist_summary.support_needed_candidate_ids,
        authenticity_review_candidate_ids=shortlist_summary.authenticity_review_candidate_ids,
        ranker_metadata={
            **get_offline_ranker_metadata(),
            "top_k_applied": top_k,
            "full_ranked_count": len(full_ranked_candidate_ids),
        },
    )
