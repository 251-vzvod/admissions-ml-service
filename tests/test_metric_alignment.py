import json

from app.config import CONFIG
from app.services.eligibility import evaluate_eligibility
from app.services.llm_parser import parse_llm_extraction_json
from app.services.preprocessing import preprocess_text_inputs
from app.services.structured_features import extract_structured_features
from app.services.text_features import extract_text_features


def test_llm_parser_supports_rubric_0_to_3_scale() -> None:
    payload = {
        "motivation_clarity": 3,
        "initiative": 2,
        "leadership_impact": 1,
        "growth_trajectory": 0,
        "resilience": 3,
        "program_fit": 2,
        "evidence_richness": 1,
        "specificity_score": 2,
        "evidence_count": 1,
        "consistency_score": 3,
        "completeness_score": 2,
        "genericness_score": 1,
        "contradiction_flag": False,
        "polished_but_empty_score": 1,
        "cross_section_mismatch_score": 2,
    }
    parsed = parse_llm_extraction_json(json.dumps(payload))
    assert parsed.motivation_clarity == 1.0
    assert round(parsed.initiative, 4) == round(2.0 / 3.0, 4)
    assert round(parsed.leadership_impact, 4) == round(1.0 / 3.0, 4)


def test_baseline_text_features_include_new_mismatch_signals() -> None:
    bundle = preprocess_text_inputs(
        {
            "motivation_letter_text": "I am passionate and motivated. I want to change the world.",
            "motivation_questions": [
                {
                    "question": "initiative",
                    "answer": "I organized a team of 6 students for 3 months and improved attendance by 20%.",
                }
            ],
            "interview_text": "I can explain concrete steps, timeline, and outcomes.",
        }
    )
    structured = extract_structured_features(structured_data={}, behavioral_signals={}, bundle=bundle)
    text = extract_text_features(bundle=bundle, structured=structured.features)

    assert "polished_but_empty_score" in text.features
    assert "cross_section_mismatch_score" in text.features
    assert 0.0 <= float(text.features["polished_but_empty_score"]) <= 1.0
    assert 0.0 <= float(text.features["cross_section_mismatch_score"]) <= 1.0


def test_eligibility_can_require_formal_materials() -> None:
    bundle = preprocess_text_inputs(
        {
            "motivation_letter_text": "I built a project and documented outcomes over 4 months with my team.",
            "motivation_questions": [{"question": "q", "answer": "Detailed answer with concrete examples and results."}],
            "interview_text": "Detailed interview response.",
        }
    )

    old_require_video = CONFIG.thresholds.require_video_presentation
    old_min_docs = CONFIG.thresholds.min_required_documents

    CONFIG.thresholds.require_video_presentation = True
    CONFIG.thresholds.min_required_documents = 1
    try:
        result = evaluate_eligibility(
            candidate_id="cand_materials_1",
            consent=True,
            bundle=bundle,
            profile={"structured_data": {"application_materials": {}}},
        )
        assert result.status == "conditionally_eligible"
        assert any(reason.startswith("missing_required_materials") for reason in result.reasons)
    finally:
        CONFIG.thresholds.require_video_presentation = old_require_video
        CONFIG.thresholds.min_required_documents = old_min_docs
