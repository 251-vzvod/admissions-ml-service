from copy import deepcopy

from app.services.pipeline import ScoringPipeline


def test_generic_text_increases_risk() -> None:
    pipeline = ScoringPipeline()

    base_payload = {
        "candidate_id": "cand_logic_1",
        "structured_data": {
            "education": {
                "english_proficiency": {"type": "ielts", "score": 7.0},
                "school_certificate": {"type": "unt", "score": 105},
            }
        },
        "text_inputs": {
            "motivation_letter_text": "I organized a team project for 4 months and improved attendance by 25%.",
            "motivation_questions": [
                {
                    "question": "initiative",
                    "answer": "I created a student workshop and documented outcomes every week.",
                }
            ],
            "interview_text": "I can explain exactly what I did and what changed after the project.",
        },
    }

    generic_payload = deepcopy(base_payload)
    generic_payload["text_inputs"]["motivation_letter_text"] = (
        "I am passionate and motivated. I want to grow and change the world. " * 10
    )

    base_result = pipeline.score_candidate(base_payload)
    generic_result = pipeline.score_candidate(generic_payload)

    assert generic_result.authenticity_risk >= base_result.authenticity_risk


def test_remove_evidence_reduces_confidence() -> None:
    pipeline = ScoringPipeline()

    rich_payload = {
        "candidate_id": "cand_logic_2",
        "text_inputs": {
            "motivation_letter_text": "I led a volunteer project with 8 students and reduced waste by 30% in 2 months.",
            "motivation_questions": [
                {"question": "example", "answer": "I built a plan, assigned roles, and reported measurable outcomes."}
            ],
            "interview_text": "I can provide detailed steps and timeline of execution.",
        },
    }

    weak_payload = deepcopy(rich_payload)
    weak_payload["text_inputs"]["motivation_letter_text"] = "I am motivated and ready to grow."
    weak_payload["text_inputs"]["motivation_questions"] = [{"question": "example", "answer": "I am passionate."}]

    rich_result = pipeline.score_candidate(rich_payload)
    weak_result = pipeline.score_candidate(weak_payload)

    assert weak_result.confidence_score <= rich_result.confidence_score
