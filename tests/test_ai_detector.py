from app.config import CONFIG
from app.services.ai_detector import detect_ai_generated_text
from app.services.authenticity import estimate_authenticity_risk
from app.services.preprocessing import preprocess_text_inputs


def test_ai_detector_disabled_by_default() -> None:
    old_enabled = CONFIG.ai_detector.enabled
    CONFIG.ai_detector.enabled = False
    bundle = preprocess_text_inputs(
        {
            "motivation_letter_text": "I built a study group and improved attendance by 30 percent over two months.",
            "interview_text": "I can explain what changed and why it worked.",
        }
    )
    try:
        result = detect_ai_generated_text(bundle)
        assert result.enabled is False
        assert result.applicable is False
        assert result.note == "disabled"
    finally:
        CONFIG.ai_detector.enabled = old_enabled


def test_ai_detector_english_applicability_with_mock(monkeypatch) -> None:
    old_enabled = CONFIG.ai_detector.enabled
    old_min_words = CONFIG.ai_detector.min_words
    try:
        CONFIG.ai_detector.enabled = True
        CONFIG.ai_detector.min_words = 20
        monkeypatch.setattr("app.services.ai_detector._detect_language", lambda text: "en")
        monkeypatch.setattr("app.services.ai_detector._predict_probability", lambda text: 0.91)

        bundle = preprocess_text_inputs(
            {
                "motivation_letter_text": (
                    "I organized a club, tracked attendance weekly, improved participation by 30 percent, "
                    "and documented what changed over two months for my classmates."
                ),
                "interview_text": "I can explain one failure, one adjustment, and the final result in detail.",
            }
        )
        result = detect_ai_generated_text(bundle)
        assert result.enabled is True
        assert result.applicable is True
        assert result.language == "en"
        assert result.probability_ai_generated is not None
        assert abs(result.probability_ai_generated - 0.91) < 1e-6
        assert result.source_results
        assert any(item.source_key == "motivation_letter_text" for item in result.source_results)
    finally:
        CONFIG.ai_detector.enabled = old_enabled
        CONFIG.ai_detector.min_words = old_min_words


def test_authenticity_risk_uses_ai_detector_as_weak_signal() -> None:
    base_features = {
        "genericness_score": 0.25,
        "evidence_count": 0.40,
        "consistency_score": 0.70,
        "polished_but_empty_score": 0.20,
        "cross_section_mismatch_score": 0.10,
        "contradiction_flag": False,
    }
    diagnostics = {"long_but_thin": False}

    without_detector = estimate_authenticity_risk(base_features, diagnostics)
    with_detector = estimate_authenticity_risk(
        base_features,
        diagnostics,
        ai_detector_result=type(
            "DetectorStub",
            (),
            {
                "enabled": True,
                "applicable": True,
                "language": "en",
                "probability_ai_generated": 0.93,
                "provider": "huggingface-inference",
                "model": "fakespot-ai/roberta-base-ai-text-detection-v1:fastest",
                "source_results": [],
                "note": "ok",
            },
        )(),
    )

    assert with_detector.authenticity_risk_raw > without_detector.authenticity_risk_raw
    assert "Auxiliary English-only AI detector assigned elevated AI-likeness; treat this only as a manual review signal." in with_detector.review_reasons
