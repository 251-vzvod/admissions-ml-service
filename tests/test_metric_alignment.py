import json

from app.config import CONFIG
from app.services.eligibility import evaluate_eligibility
from app.services.llm_parser import parse_llm_extraction_json
from app.services.preprocessing import preprocess_text_inputs
from app.services.structured_features import extract_structured_features
from app.services.text_features import extract_text_features


def test_llm_parser_supports_claim_to_evidence_schema() -> None:
    payload = {
        "top_strength_signals": [
            {
                "claim": "helped peer with coursework",
                "source": "motivation_questions",
                "snippet": "я после школы оставался с ним и мы вместе разбирались",
            }
        ],
        "main_gap_signals": [
            {
                "claim": "missing quantified outcomes",
                "source": "motivation_letter_text",
                "snippet": "особо своих проектов не было",
            }
        ],
        "uncertainties": [
            {
                "claim": "insufficient leadership scope evidence",
                "source": "interview_text",
                "snippet": "я редко бываю лидером",
            }
        ],
        "evidence_spans": [
            {
                "dimension": "initiative",
                "source": "motivation_questions",
                "text": "я после школы оставался с ним и мы вместе разбирались",
            }
        ],
        "extractor_rationale": "Claims are grounded in direct episodes from Q/A and interview.",
    }
    parsed = parse_llm_extraction_json(json.dumps(payload))
    assert parsed.top_strength_signals[0].claim == "helped peer with coursework"
    assert parsed.top_strength_signals[0].source == "motivation_questions"
    assert parsed.evidence_spans[0].dimension == "initiative"


def test_llm_parser_accepts_alias_keys_and_source_values() -> None:
    payload = {
        "top_strengths": [
            {
                "claim": "strong initiative evidence",
                "source": "interview",
                "snippet": "I coordinated the team and tracked weekly outcomes.",
            }
        ],
        "main_gaps": [
            {
                "claim": "missing long-term impact evidence",
                "source": "questions",
                "snippet": "We did not yet run a 12-month follow-up.",
            }
        ],
        "uncertainty_signals": [
            {
                "claim": "insufficient evidence for scale",
                "source": "motivation_letter",
                "snippet": "I plan to scale later.",
            }
        ],
        "evidence": [
            {
                "type": "initiative",
                "source": "video_interview",
                "snippet": "We assigned roles and monitored attendance weekly.",
            }
        ],
        "rationale": "Normalized alias response.",
    }

    parsed = parse_llm_extraction_json(json.dumps(payload))
    assert parsed.top_strength_signals[0].source == "interview_text"
    assert parsed.main_gap_signals[0].source == "motivation_questions"
    assert parsed.uncertainties[0].source == "motivation_letter_text"
    assert parsed.evidence_spans[0].source == "video_interview_transcript_text"
    assert parsed.evidence_spans[0].text.startswith("We assigned roles")


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
