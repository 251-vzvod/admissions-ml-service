from app.services.eligibility import evaluate_eligibility
from app.services.preprocessing import preprocess_text_inputs


def test_incomplete_application_status() -> None:
    bundle = preprocess_text_inputs(
        {
            "motivation_letter_text": "",
            "motivation_questions": [],
            "interview_text": "short",
        }
    )
    result = evaluate_eligibility(candidate_id="cand_x", consent=True, bundle=bundle)
    assert result.status in {"incomplete_application", "conditionally_eligible"}


def test_invalid_without_candidate_id() -> None:
    bundle = preprocess_text_inputs(
        {
            "motivation_letter_text": "I led a club project and learned from setbacks.",
            "motivation_questions": [],
            "interview_text": "",
        }
    )
    result = evaluate_eligibility(candidate_id="", consent=True, bundle=bundle)
    assert result.status == "invalid"
