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


def test_hidden_potential_outscores_polished_but_thin_case() -> None:
    hidden_payload = {
        "candidate_id": "cand_hidden_signal_001",
        "text_inputs": {
            "motivation_letter_text": (
                "When younger students in my school started skipping science club because they felt behind, "
                "I asked a teacher for one classroom on Saturdays and began a small peer group. "
                "The first version failed because the tasks were too hard and attendance dropped, "
                "so I split the group by level, added weekly feedback, and changed the materials. "
                "By the end of the semester twelve students were attending regularly and several joined olympiad preparation. "
                "I learned that leadership is building a system that keeps working after the first failure."
            ),
            "motivation_questions": [
                {
                    "question": "What changed in you through a difficult period?",
                    "answer": (
                        "When my family lost stable income, I had to study, help at home, and continue supporting younger students. "
                        "I became more disciplined and less afraid of responsibility."
                    ),
                }
            ],
            "interview_text": (
                "I can explain exactly what failed in the first version, what feedback I collected, and what changed afterward."
            ),
        },
        "consent": True,
    }
    polished_payload = {
        "candidate_id": "cand_polished_thin_001",
        "text_inputs": {
            "motivation_letter_text": (
                "I believe in transformative leadership, social innovation, and lifelong growth. "
                "My journey has prepared me to contribute meaningfully to a world-class community of changemakers. "
                "I am deeply motivated to collaborate, inspire, and create scalable impact through education and entrepreneurship."
            ),
            "motivation_questions": [
                {
                    "question": "Why this program?",
                    "answer": (
                        "The program matches my aspirations, values, and vision for sustainable impact. "
                        "I am confident that it will help me unlock my full potential."
                    ),
                }
            ],
            "interview_text": "I care about leadership, innovation, and social progress.",
        },
        "consent": True,
    }

    hidden_response = client.post("/score", json=hidden_payload)
    polished_response = client.post("/score", json=polished_payload)

    assert hidden_response.status_code == 200
    assert polished_response.status_code == 200

    hidden_result = hidden_response.json()
    polished_result = polished_response.json()

    assert hidden_result["hidden_potential_score"] > polished_result["hidden_potential_score"]
    assert hidden_result["trajectory_score"] > polished_result["trajectory_score"]
    assert "Hidden potential" in hidden_result["committee_cohorts"]
