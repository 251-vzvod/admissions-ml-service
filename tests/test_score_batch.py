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


def test_batch_offline_ranking_prefers_growth_signal_over_polished_thin_profile() -> None:
    payload = {
        "candidates": [
            {
                "candidate_id": "cand_batch_hidden_001",
                "text_inputs": {
                    "motivation_letter_text": (
                        "I started a Saturday study group for younger students, changed the format after the first month failed, "
                        "and tracked who returned each week."
                    ),
                    "motivation_questions": [
                        {
                            "question": "What changed in you?",
                            "answer": (
                                "I learned to collect feedback, adapt the plan, and keep responsibility when the first version did not work."
                            ),
                        }
                    ],
                    "interview_text": "I can explain what failed first, what I changed, and what improved afterward.",
                },
                "consent": True,
            },
            {
                "candidate_id": "cand_batch_polished_001",
                "text_inputs": {
                    "motivation_letter_text": (
                        "I believe in transformative leadership, innovation, and lifelong growth. "
                        "I want to contribute to a world-class community of changemakers."
                    ),
                    "motivation_questions": [
                        {
                            "question": "Why this program?",
                            "answer": "The program matches my aspirations and will unlock my potential for impact.",
                        }
                    ],
                    "interview_text": "I care about leadership and meaningful impact.",
                },
                "consent": True,
            },
        ]
    }

    response = client.post("/score/batch", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert result["ranked_candidate_ids"][0] == "cand_batch_hidden_001"


def test_rank_endpoint_returns_expected_contract() -> None:
    payload = {
        "candidates": [
            {
                "candidate_id": "cand_rank_001",
                "text_inputs": {
                    "motivation_letter_text": "I led a peer tutoring project and improved attendance in my neighborhood.",
                    "interview_text": "I can explain what I changed after early failures and what improved.",
                },
                "consent": True,
            },
            {
                "candidate_id": "cand_rank_002",
                "text_inputs": {
                    "motivation_letter_text": "I want mentorship and I already mentor younger students weekly.",
                    "interview_text": "I want to build practical community projects with measurable outcomes.",
                },
                "consent": True,
            },
        ]
    }

    response = client.post("/rank", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert result["count"] == 2
    assert "scoring_run_id" in result
    assert "scoring_version" in result
    assert "ranked_candidate_ids" in result
    assert "shortlist_candidate_ids" in result
    assert "hidden_potential_candidate_ids" in result
    assert "support_needed_candidate_ids" in result
    assert "authenticity_review_candidate_ids" in result
    assert "ranker_metadata" in result
    assert "version" in result["ranker_metadata"]
    assert "feature_count" in result["ranker_metadata"]
    assert "results" not in result


def test_rank_endpoint_is_deterministic_for_same_payload() -> None:
    payload = {
        "candidates": [
            {
                "candidate_id": "cand_rank_det_hidden",
                "text_inputs": {
                    "motivation_letter_text": (
                        "I started a Saturday study group, changed the format after it failed, and tracked who returned weekly."
                    ),
                    "interview_text": "I can describe exactly what failed first and what I changed to improve outcomes.",
                },
                "consent": True,
            },
            {
                "candidate_id": "cand_rank_det_polished",
                "text_inputs": {
                    "motivation_letter_text": "I believe in leadership and impact and want to join a world-class network.",
                    "interview_text": "I care about potential and growth.",
                },
                "consent": True,
            },
        ]
    }

    first = client.post("/rank", json=payload)
    second = client.post("/rank", json=payload)

    assert first.status_code == 200
    assert second.status_code == 200

    first_payload = first.json()
    second_payload = second.json()

    assert first_payload["count"] == second_payload["count"] == 2
    assert first_payload["scoring_version"] == second_payload["scoring_version"]
    assert first_payload["ranked_candidate_ids"] == second_payload["ranked_candidate_ids"]
    assert first_payload["shortlist_candidate_ids"] == second_payload["shortlist_candidate_ids"]
    assert first_payload["hidden_potential_candidate_ids"] == second_payload["hidden_potential_candidate_ids"]
    assert first_payload["support_needed_candidate_ids"] == second_payload["support_needed_candidate_ids"]
    assert first_payload["authenticity_review_candidate_ids"] == second_payload["authenticity_review_candidate_ids"]
    assert first_payload["ranker_metadata"] == second_payload["ranker_metadata"]
