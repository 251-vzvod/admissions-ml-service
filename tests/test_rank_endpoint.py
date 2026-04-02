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
        "returned_count",
        "ranked_candidate_ids",
        "ranked_candidates",
        "shortlist_candidate_ids",
        "hidden_potential_candidate_ids",
        "support_needed_candidate_ids",
        "authenticity_review_candidate_ids",
        "ranker_metadata",
    }

    assert expected_keys.issubset(result.keys())
    assert result["count"] == len(payload["candidates"])
    assert result["returned_count"] == len(payload["candidates"])

    ranked_ids = result["ranked_candidate_ids"]
    ranked_candidates = result["ranked_candidates"]
    input_ids = [candidate["candidate_id"] for candidate in payload["candidates"]]

    assert len(ranked_ids) == len(input_ids)
    assert len(set(ranked_ids)) == len(ranked_ids)
    assert set(ranked_ids) == set(input_ids)
    assert len(ranked_candidates) == len(input_ids)
    assert [item["candidate_id"] for item in ranked_candidates] == ranked_ids
    assert [item["rank_position"] for item in ranked_candidates] == [1, 2, 3]


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


def test_rank_endpoint_supports_top_k_without_changing_underlying_order() -> None:
    payload = {
        "candidates": [
            {
                "candidate_id": "cand_rank_topk_001",
                "text_inputs": {
                    "motivation_letter_text": (
                        "I run a small weekend tutoring group, track attendance, and changed the format after low turnout."
                    ),
                    "interview_text": "I can explain what did not work first and what improved after changes.",
                },
                "consent": True,
            },
            {
                "candidate_id": "cand_rank_topk_002",
                "text_inputs": {
                    "motivation_letter_text": "I want to grow, contribute, and join a strong academic community.",
                    "interview_text": "I care about leadership and impact.",
                },
                "consent": True,
            },
            {
                "candidate_id": "cand_rank_topk_003",
                "text_inputs": {
                    "motivation_letter_text": (
                        "I helped my school club continue after two organizers left and documented the new process."
                    ),
                    "interview_text": "I can describe the handover steps and what I learned from them.",
                },
                "consent": True,
            },
        ]
    }

    full_response = client.post("/rank", json=payload)
    top_response = client.post("/rank?top_k=2", json=payload)

    assert full_response.status_code == 200
    assert top_response.status_code == 200

    full_payload = full_response.json()
    top_payload = top_response.json()

    assert full_payload["count"] == 3
    assert top_payload["count"] == 3
    assert full_payload["returned_count"] == 3
    assert top_payload["returned_count"] == 2
    assert top_payload["ranked_candidate_ids"] == full_payload["ranked_candidate_ids"][:2]
    assert [item["candidate_id"] for item in top_payload["ranked_candidates"]] == top_payload["ranked_candidate_ids"]
    assert top_payload["ranker_metadata"]["top_k_applied"] == 2
    assert top_payload["ranker_metadata"]["full_ranked_count"] == 3
