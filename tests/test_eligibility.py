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


def test_multi_source_candidate_can_be_eligible() -> None:
    bundle = preprocess_text_inputs(
        {
            "motivation_letter_text": (
                "I organized a school initiative, kept weekly notes, and reflected on how repeated small improvements "
                "changed both the project and my goals for the future."
            ),
            "motivation_questions": [
                {
                    "question": "What did you learn?",
                    "answer": "I learned to revise the plan after setbacks, gather feedback, and explain concrete outcomes."
                }
            ],
            "interview_text": (
                "I can explain the concrete actions I took, what worked, what failed, and what I would do differently next time."
            ),
        }
    )
    result = evaluate_eligibility(candidate_id="cand_ok", consent=True, bundle=bundle)
    assert result.status == "eligible"
