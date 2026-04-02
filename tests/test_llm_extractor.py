from app.services.llm_extractor import extract_explainability_with_llm
from app.services.llm_client import LLMResponse
from app.services.preprocessing import preprocess_text_inputs
from app.services.llm_prompts import PROMPT_VERSION


class FakeClient:
    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self.requests: list[object] = []

    def complete(self, request: object) -> LLMResponse:
        self.requests.append(request)
        content = self._responses.pop(0)
        return LLMResponse(
            content=content,
            provider="openai-compatible",
            model="llama3.1:8b",
            latency_ms=123,
        )


def test_llm_extractor_repairs_wrapper_answer_response(monkeypatch) -> None:
    bundle = preprocess_text_inputs(
        {
            "motivation_letter_text": "I organized study notes for classmates and kept improving the format.",
            "motivation_questions": [
                {
                    "question": "What changed after a hard period?",
                    "answer": "I became more responsible and helped younger students prepare for exams.",
                }
            ],
        }
    )
    fake_client = FakeClient(
        [
            '{"answer":"Candidate shows initiative and growth, but evidence is lightly quantified."}',
            """
            {
              "rubric_assessment": {
                "leadership_potential": 4,
                "growth_trajectory": 4,
                "motivation_authenticity": 4,
                "evidence_strength": 3,
                "hidden_potential_hint": 4,
                "authenticity_review_needed": "low"
              },
              "top_strength_signals": [
                {
                  "claim": "Shows initiative through peer support",
                  "source": "motivation_letter_text",
                  "snippet": "I organized study notes for classmates and kept improving the format."
                }
              ],
              "main_gap_signals": [
                {
                  "claim": "Impact remains lightly quantified",
                  "source": "motivation_questions",
                  "snippet": "helped younger students prepare for exams"
                }
              ],
              "uncertainty_signals": [],
              "evidence_spans": [
                {
                  "dimension": "initiative",
                  "source": "motivation_letter_text",
                  "text": "I organized study notes for classmates and kept improving the format."
                }
              ],
              "committee_follow_up_question": "What changed after you started helping classmates?",
              "extractor_rationale": "Grounded in repeated peer-support actions."
            }
            """,
        ]
    )

    monkeypatch.setattr("app.services.llm_extractor.build_llm_client", lambda **_: fake_client)

    result = extract_explainability_with_llm(
        bundle=bundle,
        deterministic_signals={"initiative": 0.7, "growth_trajectory": 0.6},
    )

    assert len(fake_client.requests) == 2
    assert result.llm_metadata["repair_attempted"] == "true"
    assert result.llm_metadata["repair_used"] == "true"
    assert result.llm_metadata["prompt_version"] == PROMPT_VERSION
    assert result.rubric_assessment["leadership_potential"] == 4
    assert result.rubric_assessment["growth_trajectory"] == 4
    assert result.authenticity_assist.available is True
    assert result.authenticity_assist.review_needed == "low"
    assert result.committee_follow_up_question == "What changed after you started helping classmates?"
    assert result.strength_claims[0]["claim"] == "Shows initiative through peer support"


def test_llm_extractor_falls_back_to_deterministic_rubric_when_repair_is_still_unstructured(monkeypatch) -> None:
    bundle = preprocess_text_inputs(
        {
            "motivation_letter_text": "I kept helping classmates and adapting my approach after setbacks.",
        }
    )
    fake_client = FakeClient(
        [
            '{"answer":"Candidate shows growth and initiative."}',
            '{"text":"Candidate is motivated but evidence is limited."}',
        ]
    )

    monkeypatch.setattr("app.services.llm_extractor.build_llm_client", lambda **_: fake_client)

    result = extract_explainability_with_llm(
        bundle=bundle,
        deterministic_signals={
            "initiative": 0.7,
            "leadership_impact": 0.55,
            "growth_trajectory": 0.8,
            "resilience": 0.75,
            "motivation_clarity": 0.65,
            "program_fit": 0.6,
            "evidence_richness": 0.3,
            "specificity_score": 0.35,
            "evidence_count": 0.25,
            "community_value_orientation": 0.5,
            "polished_but_empty_score": 0.2,
            "cross_section_mismatch_score": 0.1,
            "consistency_score": 0.8,
            "contradiction_flag": False,
        },
    )

    assert len(fake_client.requests) == 2
    assert result.llm_metadata["repair_attempted"] == "true"
    assert result.llm_metadata["repair_used"] == "false"
    assert result.llm_metadata["deterministic_rubric_fallback"] == "true"
    assert result.llm_metadata["prompt_version"] == PROMPT_VERSION
    assert result.rubric_assessment["leadership_potential"] >= 3
    assert result.rubric_assessment["growth_trajectory"] == 5
    assert result.rubric_assessment["authenticity_review_needed"] == "low"
    assert result.authenticity_assist.available is False
