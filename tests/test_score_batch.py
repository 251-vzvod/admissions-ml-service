import json
from pathlib import Path

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_score_batch() -> None:
    data_path = Path("data/candidates.json")
    data = json.loads(data_path.read_text(encoding="utf-8"))
    payload = {"candidates": data["candidates"][:2]}

    response = client.post("/score/batch", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert result["count"] == 2
    assert len(result["results"]) == 2
    assert all("recommendation" in item for item in result["results"])
