from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_score_single_candidate() -> None:
    payload = {
        "candidate_id": "cand_test_001",
        "structured_data": {
            "education": {
                "english_proficiency": {"type": "ielts", "score": 7.0},
                "school_certificate": {"type": "unt", "score": 110},
            }
        },
        "text_inputs": {
            "motivation_letter_text": "I started a student coding club and improved participation by 30% in 3 months.",
            "motivation_questions": [
                {
                    "question": "Why this program?",
                    "answer": "Because I want to build social impact projects and I already organize peer workshops.",
                }
            ],
            "interview_text": "I led a volunteer event and documented outcomes weekly.",
        },
    }

    response = client.post("/score", json=payload)
    assert response.status_code == 200

    result = response.json()
    assert result["candidate_id"] == "cand_test_001"
    assert result["extraction_mode"] == "deterministic_scoring"
    assert "extractor_version" in result
    assert 0 <= result["merit_score"] <= 100
    assert 0 <= result["confidence_score"] <= 100
    assert 0 <= result["authenticity_risk"] <= 100
    assert result["recommendation"] in {
        "review_priority",
        "standard_review",
        "manual_review_required",
        "insufficient_evidence",
        "incomplete_application",
        "invalid",
    }


def test_score_single_candidate_with_video_transcript_only() -> None:
    payload = {
        "candidate_id": "cand_test_video_001",
        "text_inputs": {
            "video_interview_transcript_text": (
                "I planned a volunteer event, coordinated three classmates, and documented tasks over two weeks. "
                "During the interview I explained what worked, what failed, and how we improved next time."
            )
        },
    }

    response = client.post("/score", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["candidate_id"] == "cand_test_video_001"
    assert result["eligibility_status"] in {"eligible", "conditionally_eligible"}
    assert 0 <= result["merit_score"] <= 100
