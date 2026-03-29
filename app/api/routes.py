"""HTTP routes for scoring service."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.config import CONFIG
from app.schemas.input import BatchScoreRequest, CandidateInput
from app.schemas.output import BatchScoreResponse, ScoreResponse
from app.services.llm_extractor import extract_explainability_with_llm
from app.services.preprocessing import preprocess_text_inputs
from app.services.pipeline import ScoringPipeline
from app.services.structured_features import extract_structured_features
from app.services.text_features import extract_text_features
from app.utils.ids import generate_scoring_run_id


router = APIRouter()
pipeline = ScoringPipeline()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "invision-u-scoring-mvp", "scoring_version": CONFIG.scoring_version}


@router.get("/config/scoring")
def get_scoring_config() -> dict[str, Any]:
    """Return scoring config required for reproducibility and transparency."""
    return {
        "scoring_version": CONFIG.scoring_version,
        "scoring_config_version": CONFIG.scoring_config_version,
        "weight_experiment_protocol_version": CONFIG.weight_experiment_protocol_version,
        "prompt_version": CONFIG.prompt_version,
        "excluded_fields": CONFIG.excluded_fields,
        "weights": {
            "merit_breakdown": CONFIG.weights.merit_breakdown,
            "confidence_components": CONFIG.weights.confidence_components,
        },
        "thresholds": asdict(CONFIG.thresholds),
        "extraction_strategy": "deterministic_features_plus_optional_llm_explainability",
        "llm_provider": CONFIG.llm.provider,
        "llm_model": CONFIG.llm.model,
        "llm_fallback_to_deterministic_extractor_on_failure": CONFIG.llm.fallback_to_baseline,
    }


@router.post("/score", response_model=ScoreResponse)
def score_candidate(candidate: CandidateInput) -> ScoreResponse:
    return pipeline.score_candidate_model(candidate)


@router.post("/score/batch", response_model=BatchScoreResponse)
def score_batch(payload: BatchScoreRequest) -> BatchScoreResponse:
    scoring_run_id = generate_scoring_run_id()
    results = [pipeline.score_candidate_model(candidate, scoring_run_id=scoring_run_id) for candidate in payload.candidates]
    return BatchScoreResponse(
        scoring_run_id=scoring_run_id,
        scoring_version=CONFIG.scoring_version,
        count=len(results),
        results=results,
    )


@router.post("/score/file", response_model=BatchScoreResponse)
async def score_file(file_path: str | None = None, upload: UploadFile | None = File(default=None)) -> BatchScoreResponse:
    """Score candidates from local JSON path or uploaded JSON file."""
    raw: dict[str, Any]

    if upload is not None:
        content = await upload.read()
        raw = json.loads(content.decode("utf-8"))
    elif file_path:
        path = Path(file_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="file_path_not_found")
        raw = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise HTTPException(status_code=400, detail="provide_file_path_or_upload")

    candidates_raw = raw.get("candidates", [])
    if not isinstance(candidates_raw, list):
        raise HTTPException(status_code=400, detail="invalid_candidates_json_format")

    candidates = [CandidateInput.model_validate(item) for item in candidates_raw]
    return score_batch(BatchScoreRequest(candidates=candidates))


@router.post("/debug/features")
def debug_features(candidate: CandidateInput) -> dict[str, Any]:
    scored = pipeline.score_candidate_model(candidate)
    return {
        "candidate_id": scored.candidate_id,
        "feature_snapshot": scored.feature_snapshot,
        "merit_breakdown": scored.merit_breakdown,
        "review_flags": scored.review_flags,
    }


@router.post("/debug/explanation")
def debug_explanation(candidate: CandidateInput) -> dict[str, Any]:
    scored = pipeline.score_candidate_model(candidate)
    return {
        "candidate_id": scored.candidate_id,
        "top_strengths": scored.top_strengths,
        "main_gaps": scored.main_gaps,
        "uncertainties": scored.uncertainties,
        "evidence_spans": [span.model_dump() for span in scored.evidence_spans],
        "explanation": scored.explanation.model_dump(),
    }


@router.post("/debug/llm-extract")
def debug_llm_extract(candidate: CandidateInput) -> dict[str, Any]:
    """Return LLM explainability artifacts for inspection."""
    text_inputs = candidate.text_inputs.model_dump(mode="python") if candidate.text_inputs else {}
    bundle = preprocess_text_inputs(text_inputs=text_inputs)

    structured_data = candidate.structured_data.model_dump(mode="python") if candidate.structured_data else {}
    behavioral_signals = candidate.behavioral_signals.model_dump(mode="python") if candidate.behavioral_signals else {}
    structured = extract_structured_features(
        structured_data=structured_data,
        behavioral_signals=behavioral_signals,
        bundle=bundle,
    )
    deterministic_text = extract_text_features(bundle=bundle, structured=structured.features)

    result = extract_explainability_with_llm(bundle, deterministic_signals=deterministic_text.features)
    return {
        "top_strength_claims": result.strength_claims,
        "main_gap_claims": result.gap_claims,
        "uncertainty_claims": result.uncertainty_claims,
        "evidence_spans": result.evidence_spans,
        "extractor_rationale": result.rationale,
        "llm_metadata": result.llm_metadata,
    }


@router.post("/debug/score-trace")
def debug_score_trace(candidate: CandidateInput) -> dict[str, Any]:
    """Return full deterministic scoring trace for auditing."""
    return pipeline.score_candidate_trace_model(candidate)
