from app.services.llm_prompts import PROMPT_VERSION, build_extraction_user_prompt
from app.services.preprocessing import preprocess_text_inputs


def test_extraction_prompt_includes_invision_context_and_evidence_first_rules() -> None:
    bundle = preprocess_text_inputs(
        {
            "motivation_letter_text": "I started a school project and kept improving it after feedback.",
            "interview_text": "I can explain what changed after the first version failed.",
        }
    )

    prompt = build_extraction_user_prompt(
        bundle=bundle,
        deterministic_signals={"initiative": 0.7, "growth_trajectory": 0.6},
    )

    assert PROMPT_VERSION == "llm-explainability-v3-invision-agenda"
    assert "inVision U" in prompt
    assert "Work evidence-first." in prompt
    assert "Do not treat deterministic_text_signals as evidence" in prompt
    assert "committee_follow_up_question" in prompt
    assert "institution_type" in prompt
    assert "trajectory over polish" in prompt
    assert "leadership means willingness to give more than take" in prompt
    assert "false positives" in prompt
