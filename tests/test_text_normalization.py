from app.services.preprocessing import preprocess_text_inputs
from app.utils.text import normalize_multiline_text, normalize_unicode_text


def test_normalize_unicode_text_repairs_common_punctuation_and_mojibake() -> None:
    raw = "I donвЂ™t think this вЂ” or this… should stay “as is”."
    normalized = normalize_unicode_text(raw)

    assert normalized == 'I don\'t think this - or this... should stay "as is".'


def test_normalize_multiline_text_preserves_paragraphs() -> None:
    raw = " First paragraph\u00a0with odd spacing. \n\n Second\tparagraph… "

    normalized = normalize_multiline_text(raw)

    assert normalized == "First paragraph with odd spacing.\n\nSecond paragraph..."


def test_preprocess_text_inputs_uses_unicode_normalization() -> None:
    bundle = preprocess_text_inputs(
        {
            "motivation_letter_text": "I donвЂ™t want polished writing - I want useful work…",
            "motivation_questions": [
                {
                    "question": "Why this program?",
                    "answer": "Because it teaches action, not just image.",
                }
            ],
        }
    )

    assert "don't" in bundle.motivation_letter_original
    assert "..." in bundle.motivation_letter_original
    assert "вЂ" not in bundle.full_text_original
