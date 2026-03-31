from fastapi.testclient import TestClient

from app.config import CONFIG
from app.main import app


client = TestClient(app)

CONFIG.llm.enabled = False


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
    assert isinstance(result["committee_cohorts"], list)
    assert isinstance(result["why_candidate_surfaced"], list)
    assert isinstance(result["what_to_verify_manually"], list)
    assert isinstance(result["suggested_follow_up_question"], str)
    assert isinstance(result["authenticity_review_reasons"], list)
    assert "ai_detector" in result
    assert 0 <= result["hidden_potential_score"] <= 100
    assert 0 <= result["support_needed_score"] <= 100
    assert 0 <= result["shortlist_priority_score"] <= 100
    assert 0 <= result["evidence_coverage_score"] <= 100
    assert 0 <= result["trajectory_score"] <= 100


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


def test_score_single_uses_application_materials_features() -> None:
    payload = {
        "candidate_id": "cand_test_materials_001",
        "structured_data": {
            "education": {
                "english_proficiency": {"type": "ielts", "score": 6.5},
                "school_certificate": {"type": "unt", "score": 105},
            },
            "application_materials": {
                "documents": ["cv.pdf", "statement.pdf"],
                "attachments": ["portfolio.pdf"],
                "portfolio_links": ["https://example.com/portfolio"],
                "video_presentation_link": "https://example.com/video",
            },
        },
        "text_inputs": {
            "motivation_letter_text": "I organized a student project with measurable outcomes and weekly reports.",
            "interview_text": "I led a team of five and tracked participation metrics over three months.",
        },
    }

    response = client.post("/score", json=payload)
    assert response.status_code == 200
    result = response.json()
    assert result["candidate_id"] == "cand_test_materials_001"
    assert result["evidence_coverage_score"] >= 0
    assert result["confidence_score"] >= 0
    assert isinstance(result["committee_cohorts"], list)
