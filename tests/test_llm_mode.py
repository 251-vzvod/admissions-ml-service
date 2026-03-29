from copy import deepcopy
import json

from app.config import CONFIG
from app.main import app
from app.services.llm_client import LLMResponse, OpenAICompatibleClient
from app.services.pipeline import ScoringPipeline
from fastapi.testclient import TestClient


client = TestClient(app)


def _valid_llm_json() -> str:
    return json.dumps(
        {
            "top_strength_signals": [
                {
                    "claim": "self-started initiative",
                    "source": "motivation_questions",
                    "snippet": "I organized a volunteer team and tracked weekly outcomes.",
                },
                {
                    "claim": "clear growth reflection",
                    "source": "interview_text",
                    "snippet": "I can explain actions, timeline, and results with concrete examples.",
                },
            ],
            "main_gap_signals": [
                {
                    "claim": "missing quantified outcomes beyond one project",
                    "source": "motivation_letter_text",
                    "snippet": "I started a school project and improved participation by 20% in two months.",
                }
            ],
            "uncertainties": [
                {
                    "claim": "insufficient breadth of leadership evidence",
                    "source": "motivation_questions",
                    "snippet": "I organized a volunteer team and tracked weekly outcomes.",
                }
            ],
            "evidence_spans": [
                {
                    "dimension": "initiative",
                    "source": "motivation_questions",
                    "text": "I started a student club and organized weekly sessions",
                }
            ],
            "extractor_rationale": "Signals are grounded in repeated action examples and reflective narrative.",
        }
    )


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


def test_hybrid_mode_default_with_fallback_metadata(monkeypatch) -> None:
    pipeline = ScoringPipeline()
    old_provider = CONFIG.llm.provider

    def _failing_complete(self, request):
        raise RuntimeError("forced_llm_failure")

    monkeypatch.setattr(OpenAICompatibleClient, "complete", _failing_complete)

    try:
        CONFIG.llm.provider = "openai"
        result = pipeline.score_candidate(_payload())
        assert result.extraction_mode == "deterministic_scoring"
        assert isinstance(result.llm_metadata, dict)
    finally:
        CONFIG.llm.provider = old_provider


def test_llm_mode_with_openai_client_patch(monkeypatch) -> None:
    pipeline = ScoringPipeline()
    old_provider = CONFIG.llm.provider

    def _fake_complete(self, request):
        return LLMResponse(content=_valid_llm_json(), provider="openai", model=request.model, latency_ms=1)

    monkeypatch.setattr(OpenAICompatibleClient, "complete", _fake_complete)

    CONFIG.llm.provider = "openai"
    try:
        result = pipeline.score_candidate(_payload())
        assert result.extraction_mode == "deterministic_scoring"
        assert result.llm_metadata is not None
        assert result.llm_metadata.get("provider") == "openai"
    finally:
        CONFIG.llm.provider = old_provider


def test_llm_parsed_extraction_keeps_deterministic_scoring(monkeypatch) -> None:
    pipeline = ScoringPipeline()
    old_provider = CONFIG.llm.provider

    def _fake_complete(self, request):
        return LLMResponse(content=_valid_llm_json(), provider="openai", model=request.model, latency_ms=1)

    monkeypatch.setattr(OpenAICompatibleClient, "complete", _fake_complete)

    CONFIG.llm.provider = "openai"
    try:
        result_a = pipeline.score_candidate(_payload())
        result_b = pipeline.score_candidate(_payload())
        assert result_a.merit_score == result_b.merit_score
        assert result_a.confidence_score == result_b.confidence_score
        assert result_a.authenticity_risk == result_b.authenticity_risk
        assert result_a.recommendation == result_b.recommendation
    finally:
        CONFIG.llm.provider = old_provider


def test_invalid_llm_json_falls_back_with_hybrid_mode(monkeypatch) -> None:
    pipeline = ScoringPipeline()
    old_provider = CONFIG.llm.provider
    old_fallback = CONFIG.llm.fallback_to_baseline

    def _invalid_complete(self, request):
        return LLMResponse(content="{invalid_json", provider="openai", model=request.model, latency_ms=1)

    monkeypatch.setattr(OpenAICompatibleClient, "complete", _invalid_complete)

    CONFIG.llm.provider = "openai"
    CONFIG.llm.fallback_to_baseline = True
    try:
        result = pipeline.score_candidate(_payload())
        assert result.extraction_mode == "deterministic_scoring"
        assert result.llm_metadata is not None
        assert "fallback_reason" in result.llm_metadata
    finally:
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


def test_sensitive_fields_do_not_affect_merit(monkeypatch) -> None:
    pipeline = ScoringPipeline()
    baseline = _payload()

    def _fake_complete(self, request):
        return LLMResponse(content=_valid_llm_json(), provider="openai", model=request.model, latency_ms=1)

    monkeypatch.setattr(OpenAICompatibleClient, "complete", _fake_complete)

    with_sensitive = deepcopy(baseline)
    with_sensitive["gender"] = "female"
    with_sensitive["income"] = "low"
    with_sensitive["citizenship"] = "KZ"
    with_sensitive["family_details"] = {"parents": "single_parent"}

    result_a = pipeline.score_candidate(baseline)
    result_b = pipeline.score_candidate(with_sensitive)

    assert result_a.merit_score == result_b.merit_score


def test_debug_llm_extract_endpoint_with_openai_patch(monkeypatch) -> None:
    old_provider = CONFIG.llm.provider

    def _fake_complete(self, request):
        return LLMResponse(content=_valid_llm_json(), provider="openai", model=request.model, latency_ms=1)

    monkeypatch.setattr(OpenAICompatibleClient, "complete", _fake_complete)

    CONFIG.llm.provider = "openai"

    payload = _payload()
    try:
        response = client.post("/debug/llm-extract", json=payload)
        assert response.status_code == 200
        parsed = response.json()
        assert "top_strength_claims" in parsed
        assert "main_gap_claims" in parsed
        assert "llm_metadata" in parsed
        assert parsed["llm_metadata"]["provider"] == "openai"
    finally:
        CONFIG.llm.provider = old_provider


def test_debug_score_trace_endpoint_returns_formula_and_components() -> None:
    payload = _payload()
    response = client.post("/debug/score-trace", json=payload)
    assert response.status_code == 200
    parsed = response.json()
    assert parsed.get("extraction_mode") == "deterministic_scoring"
    assert "score_trace" in parsed
    assert "formulas" in parsed["score_trace"]
    assert "components" in parsed["score_trace"]
