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


def test_hidden_potential_is_not_punished_below_polished_low_signal_profile() -> None:
    pipeline = ScoringPipeline()

    hidden_potential_payload = {
        "candidate_id": "cand_hidden_potential",
        "text_inputs": {
            "motivation_letter_text": (
                "My father lost his job two years ago, so I started helping my younger brother study every evening. "
                "Later I organized shared notes for classmates before exams and kept improving them after feedback."
            ),
            "motivation_questions": [
                {
                    "question": "Tell us about your initiative.",
                    "answer": (
                        "Nobody asked me to do it. I made a simple study routine for three classmates, tracked what confused them, "
                        "and changed the plan each week when something failed."
                    ),
                }
            ],
            "interview_text": (
                "I am not the strongest speaker, but I can explain what I changed, why it helped, and what I learned from setbacks."
            ),
        },
    }

    polished_low_signal_payload = {
        "candidate_id": "cand_polished_low_signal",
        "text_inputs": {
            "motivation_letter_text": (
                "I am passionate, ambitious, and eager to grow into the best version of myself. "
                "I dream of changing the world through interdisciplinary teamwork and meaningful impact."
            ),
            "motivation_questions": [
                {
                    "question": "Tell us about your initiative.",
                    "answer": "I am deeply motivated to make a difference and bring positive energy to any environment.",
                }
            ],
            "interview_text": "I strongly believe this program will unlock my potential and shape my future.",
        },
    }

    hidden_result = pipeline.score_candidate(hidden_potential_payload)
    polished_result = pipeline.score_candidate(polished_low_signal_payload)

    assert hidden_result.merit_score >= polished_result.merit_score
