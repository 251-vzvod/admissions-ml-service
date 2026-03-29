from copy import deepcopy

from app.config import CONFIG
from app.main import app
from app.services.pipeline import ScoringPipeline
from fastapi.testclient import TestClient


client = TestClient(app)


def _payload() -> dict:
    return {
        "candidate_id": "cand_llm_001",
        "text_inputs": {
            "motivation_letter_text": "I started a school project and improved participation by 20% in two months.",
            "motivation_questions": [
                {
                    "question": "initiative",
                    "answer": "I organized a volunteer team and tracked weekly outcomes.",
                }
            ],
            "interview_text": "I can explain actions, timeline, and results with concrete examples.",
        },
    }


def test_baseline_mode_works_without_llm() -> None:
    pipeline = ScoringPipeline()
    old_enable = CONFIG.llm.enable_llm
    CONFIG.llm.enable_llm = False
    try:
        result = pipeline.score_candidate(_payload())
        assert result.extraction_mode == "baseline"
        assert result.llm_metadata is None
    finally:
        CONFIG.llm.enable_llm = old_enable


def test_llm_mode_with_mock_provider() -> None:
    pipeline = ScoringPipeline()
    old_enable = CONFIG.llm.enable_llm
    old_provider = CONFIG.llm.provider

    CONFIG.llm.enable_llm = True
    CONFIG.llm.provider = "mock"
    try:
        result = pipeline.score_candidate(_payload())
        assert result.extraction_mode == "llm"
        assert result.llm_metadata is not None
        assert result.llm_metadata.get("provider") == "mock"
    finally:
        CONFIG.llm.enable_llm = old_enable
        CONFIG.llm.provider = old_provider


def test_llm_parsed_extraction_keeps_deterministic_scoring() -> None:
    pipeline = ScoringPipeline()
    old_enable = CONFIG.llm.enable_llm
    old_provider = CONFIG.llm.provider

    CONFIG.llm.enable_llm = True
    CONFIG.llm.provider = "mock"
    try:
        result_a = pipeline.score_candidate(_payload())
        result_b = pipeline.score_candidate(_payload())
        assert result_a.merit_score == result_b.merit_score
        assert result_a.confidence_score == result_b.confidence_score
        assert result_a.authenticity_risk == result_b.authenticity_risk
        assert result_a.recommendation == result_b.recommendation
    finally:
        CONFIG.llm.enable_llm = old_enable
        CONFIG.llm.provider = old_provider


def test_invalid_llm_json_falls_back_to_baseline() -> None:
    pipeline = ScoringPipeline()
    old_enable = CONFIG.llm.enable_llm
    old_provider = CONFIG.llm.provider
    old_fallback = CONFIG.llm.fallback_to_baseline

    CONFIG.llm.enable_llm = True
    CONFIG.llm.provider = "mock_invalid"
    CONFIG.llm.fallback_to_baseline = True
    try:
        result = pipeline.score_candidate(_payload())
        assert result.extraction_mode == "baseline"
        assert result.llm_metadata is not None
        assert "fallback_reason" in result.llm_metadata
    finally:
        CONFIG.llm.enable_llm = old_enable
        CONFIG.llm.provider = old_provider
        CONFIG.llm.fallback_to_baseline = old_fallback


def test_high_risk_does_not_directly_reject_candidate() -> None:
    pipeline = ScoringPipeline()
    payload = _payload()
    payload["text_inputs"]["motivation_letter_text"] = "I am passionate and motivated. I want to grow and change the world. " * 25

    result = pipeline.score_candidate(payload)
    assert result.recommendation in {
        "standard_review",
        "manual_review_required",
        "insufficient_evidence",
        "review_priority",
    }
    assert result.recommendation not in {"invalid", "incomplete_application"}


def test_sensitive_fields_do_not_affect_merit() -> None:
    pipeline = ScoringPipeline()
    baseline = _payload()

    with_sensitive = deepcopy(baseline)
    with_sensitive["gender"] = "female"
    with_sensitive["income"] = "low"
    with_sensitive["citizenship"] = "KZ"
    with_sensitive["family_details"] = {"parents": "single_parent"}

    result_a = pipeline.score_candidate(baseline)
    result_b = pipeline.score_candidate(with_sensitive)

    assert result_a.merit_score == result_b.merit_score


def test_debug_llm_extract_endpoint_with_mock() -> None:
    old_enable = CONFIG.llm.enable_llm
    old_provider = CONFIG.llm.provider
    CONFIG.llm.enable_llm = True
    CONFIG.llm.provider = "mock"

    payload = _payload()
    try:
        response = client.post("/debug/llm-extract", json=payload)
        assert response.status_code == 200
        parsed = response.json()
        assert "features" in parsed
        assert "llm_metadata" in parsed
        assert parsed["llm_metadata"]["provider"] == "mock"
    finally:
        CONFIG.llm.enable_llm = old_enable
        CONFIG.llm.provider = old_provider
