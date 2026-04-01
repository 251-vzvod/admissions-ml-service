from fastapi.testclient import TestClient

from app.config import CONFIG
from app.main import app


client = TestClient(app)

CONFIG.llm.enabled = False


def test_rank_endpoint_response_contract_and_ranked_ids_integrity() -> None:
    payload = {
        "candidates": [
            {
                "candidate_id": "cand_rank_contract_001",
                "text_inputs": {
                    "motivation_letter_text": (
                        "I run peer tutoring sessions twice a week and track attendance improvements."
                    ),
                    "interview_text": (
                        "I can explain what failed in the first format and how the updated plan improved outcomes."
                    ),
                },
                "consent": True,
            },
            {
                "candidate_id": "cand_rank_contract_002",
                "text_inputs": {
                    "motivation_letter_text": (
                        "I coordinate volunteers for weekend activities and report monthly progress to teachers."
                    ),
                    "interview_text": "I want mentoring to scale what already works in my community.",
                },
                "consent": True,
            },
            {
                "candidate_id": "cand_rank_contract_003",
                "text_inputs": {
                    "motivation_letter_text": (
                        "I improved school club participation by piloting smaller groups and collecting feedback."
                    ),
                    "interview_text": "I can show concrete before and after participation numbers.",
                },
                "consent": True,
            },
        ]
    }

    response = client.post("/rank", json=payload)

    assert response.status_code == 200

    result = response.json()
    expected_keys = {
        "scoring_run_id",
        "scoring_version",
        "count",
        "ranked_candidate_ids",
        "shortlist_candidate_ids",
        "hidden_potential_candidate_ids",
        "support_needed_candidate_ids",
        "authenticity_review_candidate_ids",
        "ranker_metadata",
    }

    assert expected_keys.issubset(result.keys())
    assert result["count"] == len(payload["candidates"])

    ranked_ids = result["ranked_candidate_ids"]
    input_ids = [candidate["candidate_id"] for candidate in payload["candidates"]]

    assert len(ranked_ids) == len(input_ids)
    assert len(set(ranked_ids)) == len(ranked_ids)
    assert set(ranked_ids) == set(input_ids)


def test_rank_endpoint_prefers_growth_rich_candidate_over_polished_thin_candidate() -> None:
    payload = {
        "candidates": [
            {
                "candidate_id": "cand_rank_growth_001",
                "text_inputs": {
                    "motivation_letter_text": (
                        "I started a Saturday study group for younger students, changed the format after month one failed, "
                        "and tracked who returned each week."
                    ),
                    "motivation_questions": [
                        {
                            "question": "What changed in you?",
                            "answer": (
                                "I learned to collect feedback, adapt the plan, and keep responsibility when the first "
                                "version did not work."
                            ),
                        }
                    ],
                    "interview_text": "I can explain what failed first, what I changed, and what improved afterward.",
                },
                "consent": True,
            },
            {
                "candidate_id": "cand_rank_polished_001",
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

    response = client.post("/rank", json=payload)

    assert response.status_code == 200
    result = response.json()

    assert result["ranked_candidate_ids"][0] == "cand_rank_growth_001"
