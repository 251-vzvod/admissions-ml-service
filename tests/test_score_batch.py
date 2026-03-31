from fastapi.testclient import TestClient

from app.config import CONFIG
from app.main import app


client = TestClient(app)

CONFIG.llm.enabled = False


def test_score_batch() -> None:
    payload = {
        "candidates": [
            {
                "candidate_id": "cand_batch_001",
                "text_inputs": {
                    "motivation_letter_text": "I organized peer study sessions and improved attendance over one semester.",
                    "interview_text": "I explained what failed in the first version and how I changed the format.",
                },
                "consent": True,
            },
            {
                "candidate_id": "cand_batch_002",
                "text_inputs": {
                    "motivation_letter_text": "I want to study because I want to help my community through practical projects.",
                    "motivation_questions": [
                        {
                            "question": "Why this program?",
                            "answer": "I need stronger mentorship and I already run a small volunteer tutoring group.",
                        }
                    ],
                },
                "consent": True,
            },
        ]
    }

    response = client.post("/score/batch", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert result["count"] == 2
    assert len(result["results"]) == 2
    assert "ranked_candidate_ids" in result
    assert "shortlist_candidate_ids" in result
    assert "hidden_potential_candidate_ids" in result
    assert "support_needed_candidate_ids" in result
    assert "authenticity_review_candidate_ids" in result
    assert all("recommendation" in item for item in result["results"])
