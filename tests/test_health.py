from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert "scoring_version" in payload


def test_scoring_config_contains_extraction_strategy_fields() -> None:
    response = client.get("/config/scoring")
    assert response.status_code == 200
    payload = response.json()
    assert "extraction_strategy" in payload
    assert "llm_provider" in payload
    assert "llm_model" in payload
    assert "llm_fallback_to_deterministic_extractor_on_failure" in payload
