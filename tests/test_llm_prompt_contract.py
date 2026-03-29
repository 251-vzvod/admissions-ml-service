from app.services.llm_prompts import SYSTEM_PROMPT, build_extraction_user_prompt
from app.services.preprocessing import preprocess_text_inputs


def test_system_prompt_enforces_evidence_contract() -> None:
    assert "You must NOT output numeric scoring fields." in SYSTEM_PROMPT
    assert "top_strength_signals: 2-3 items in claim->evidence format" in SYSTEM_PROMPT
    assert "main_gap_signals: 2-3 items in claim->evidence format" in SYSTEM_PROMPT
    assert "uncertainties: 1-3 items in claim->evidence format" in SYSTEM_PROMPT
    assert "Every claim must include source and direct snippet from input text" in SYSTEM_PROMPT
    assert "If motivation_questions are present, include at least one evidence span from motivation_questions" in SYSTEM_PROMPT


def test_user_prompt_includes_questions_and_answers() -> None:
    bundle = preprocess_text_inputs(
        {
            "motivation_letter_text": "I want to learn by building projects.",
            "motivation_questions": [
                {
                    "question": "What did you build?",
                    "answer": "I organized a school event and tracked attendance weekly.",
                }
            ],
            "interview_text": "I can describe exact actions and outcomes.",
        }
    )

    user_prompt = build_extraction_user_prompt(bundle)

    assert '"motivation_questions"' in user_prompt
    assert '"question": "What did you build?"' in user_prompt
    assert '"answer": "I organized a school event and tracked attendance weekly."' in user_prompt
