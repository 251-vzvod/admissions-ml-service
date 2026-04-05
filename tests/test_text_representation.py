from app.services.preprocessing import preprocess_text_inputs
from app.services.text_representation import TextRepresentationConfig, build_text_representation


def test_text_representation_builds_source_aware_feature_map_with_hash_backend() -> None:
    bundle = preprocess_text_inputs(
        {
            "motivation_letter_text": (
                "I started a study group for younger students and kept improving it after feedback.\n\n"
                "Later we turned it into regular peer support."
            ),
            "motivation_questions": [
                {
                    "question": "Why do you want to study here?",
                    "answer": "I want to build things that improve opportunities for other people in my city.",
                }
            ],
            "interview_text": (
                "I noticed that some classmates had no one to explain math tasks to them, so I organized short weekly sessions."
            ),
        }
    )

    result = build_text_representation(
        bundle,
        config=TextRepresentationConfig(backend="hash"),
    )

    assert result.diagnostics["active_backend"] == "hash"
    assert result.feature_map["repr_source_coverage"] > 0.0
    assert result.feature_map["repr_interview_motivation_priority"] >= 0.0
    assert result.feature_map["repr_qa_service_priority"] >= 0.0
    assert result.feature_map["repr_best_chunk_strength"] > 0.0
    assert result.feature_map["repr_essay_growth_top"] >= 0.0
    assert result.feature_map["repr_interview_service_top"] >= 0.0
