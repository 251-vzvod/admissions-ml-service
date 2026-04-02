from app.config import CONFIG
from app.main import app
from app.services.pipeline import ScoringPipeline
from fastapi.testclient import TestClient


client = TestClient(app)


def test_review_routing_sidecar_runs_in_shadow_mode_without_public_field_leak() -> None:
    previous = CONFIG.review_routing_sidecar.enabled
    CONFIG.review_routing_sidecar.enabled = True
    try:
        payload = {
            "candidate_id": "cand_shadow_sidecar_001",
            "text_inputs": {
                "motivation_letter_text": (
                    "I want to study and help my community, but my examples are still limited and not very well organized."
                ),
                "motivation_questions": [
                    {
                        "question": "What support may you need?",
                        "answer": "I may need academic support and help expressing my ideas more clearly.",
                    }
                ],
                "interview_text": "",
            },
        }

        response = client.post("/score", json=payload)
        assert response.status_code == 200
        public_result = response.json()
        assert "review_routing_shadow" not in public_result

        pipeline = ScoringPipeline()
        internal_result = pipeline.score_candidate(payload, enable_llm_explainability=False)
        assert internal_result.review_routing_shadow is not None
        assert internal_result.review_routing_shadow.enabled is True

        trace = pipeline.score_candidate_trace(payload)
        assert "review_routing_shadow" in trace
        assert trace["review_routing_shadow"]["enabled"] is True
    finally:
        CONFIG.review_routing_sidecar.enabled = previous
