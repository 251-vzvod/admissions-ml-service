from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "scoring_version" in payload


def test_score_route_uses_config_llm_enabled_flag(monkeypatch) -> None:
    from app.config import CONFIG
    from app.services.pipeline import ScoringPipeline

    captured: dict[str, object] = {}
    original = ScoringPipeline.score_candidate_model

    def _wrapped(self, model, scoring_run_id=None, enable_llm_explainability=True):
        captured["enable_llm_explainability"] = enable_llm_explainability
        return original(
            self,
            model,
            scoring_run_id=scoring_run_id,
            enable_llm_explainability=enable_llm_explainability,
        )

    payload = {
        "candidate_id": "cand_health_001",
        "text_inputs": {"motivation_letter_text": "I started a small school initiative and kept working on it."},
        "consent": True,
    }

    old_enabled = CONFIG.llm.enabled
    monkeypatch.setattr(ScoringPipeline, "score_candidate_model", _wrapped)
    CONFIG.llm.enabled = False
    try:
        response = client.post("/score", json=payload)
        assert response.status_code == 200
        assert captured["enable_llm_explainability"] is False
    finally:
        CONFIG.llm.enabled = old_enabled
