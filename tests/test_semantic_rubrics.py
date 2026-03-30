from app.services.pipeline import ScoringPipeline
from app.services.preprocessing import preprocess_text_inputs
from app.services.semantic_rubrics import extract_semantic_rubric_features


def test_semantic_rubrics_extract_dimension_scores() -> None:
    bundle = preprocess_text_inputs(
        {
            "motivation_letter_text": (
                "I organized classmates to create shared study notes after my father lost his job and I had to balance "
                "school with family responsibilities. We kept improving the notes after feedback each week."
            ),
            "motivation_questions": [
                {
                    "question": "Why this program?",
                    "answer": (
                        "I want a project-based university because I learned that working with people and solving real "
                        "problems matters more to me than just grades."
                    ),
                }
            ],
            "interview_text": "I can describe what failed first, how I adapted the plan, and what changed in my classmates' results.",
        }
    )

    result = extract_semantic_rubric_features(bundle)

    assert result.features["semantic_leadership_potential"] > 0
    assert result.features["semantic_growth_trajectory"] > 0
    assert result.features["semantic_motivation_authenticity"] > 0
    assert "leadership_potential" in result.evidence


def test_pipeline_returns_semantic_scores() -> None:
    pipeline = ScoringPipeline()
    payload = {
        "candidate_id": "cand_semantic",
        "text_inputs": {
            "motivation_letter_text": (
                "I stood up for a classmate, then organized a small peer support group and tracked what helped people most."
            ),
            "motivation_questions": [
                {
                    "question": "Tell us about growth.",
                    "answer": "After failing at first, I changed the plan, asked for feedback, and improved it every week.",
                }
            ],
            "interview_text": "This program fits because I want to build projects for my community, not only study theory.",
        },
    }

    result = pipeline.score_candidate(payload)

    assert result.semantic_rubric_scores["leadership_potential"] > 0
    assert result.semantic_rubric_scores["growth_trajectory"] > 0
