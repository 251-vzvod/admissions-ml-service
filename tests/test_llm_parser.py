from app.services.llm_parser import parse_llm_extraction_json


def test_llm_parser_accepts_calibration_adjudication_schema() -> None:
    raw = """
    {
      "candidate_id": "cand_calib_001",
      "human_review": {
        "recommendation": "review_priority",
        "shortlist_band": true,
        "hidden_potential_band": true,
        "support_needed_band": false,
        "authenticity_review_band": false,
        "notes": "Strong action and growth signal with imperfect polish."
      },
      "rubric": {
        "leadership_through_action": 4,
        "growth_trajectory": 5,
        "community_orientation": 4,
        "project_based_readiness": 4,
        "motivation_groundedness": 4,
        "evidence_strength": 3,
        "shortlist_priority": 4
      },
      "evidence_bullets": [
        "Started a peer tutoring group and changed the format after the first version failed.",
        "Motivation is linked to helping younger students, not only personal advancement."
      ],
      "uncertainties": [
        "Outcomes are promising but only lightly quantified."
      ]
    }
    """

    parsed = parse_llm_extraction_json(raw)

    assert parsed.candidate_id == "cand_calib_001"
    assert parsed.human_review is not None
    assert parsed.human_review.recommendation == "review_priority"
    assert parsed.human_review.shortlist_band is True
    assert parsed.human_review.hidden_potential_band is True
    assert parsed.rubric["leadership_through_action"] == 4
    assert parsed.rubric["growth_trajectory"] == 5
    assert parsed.rubric["shortlist_priority"] == 4
    assert len(parsed.evidence_bullets) == 2
    assert parsed.uncertainties == ["Outcomes are promising but only lightly quantified."]


def test_llm_parser_normalizes_calibration_aliases_and_bounds() -> None:
    raw = """
    {
      "candidate_id": "cand_calib_002",
      "human_review": {
        "recommendation": "MANUAL_REVIEW_REQUIRED",
        "shortlist_band": "yes",
        "hidden_potential_band": "no",
        "needs_support_flag": "1",
        "authenticity_review_needed": "HIGH",
        "notes": "  Needs verification.  "
      },
      "rubric": {
        "leadership_potential": 7,
        "growth_trajectory": 0,
        "community_orientation": "4",
        "readiness": 3,
        "motivation_authenticity": "2",
        "evidence_strength": "6",
        "shortlist_priority": "0"
      },
      "evidence_bullets": ["  Clear initiative.  ", "", null],
      "uncertainties": ["  Outcome metrics are thin.  "]
    }
    """

    parsed = parse_llm_extraction_json(raw)

    assert parsed.human_review is not None
    assert parsed.human_review.recommendation == "manual_review_required"
    assert parsed.human_review.shortlist_band is True
    assert parsed.human_review.hidden_potential_band is False
    assert parsed.human_review.support_needed_band is True
    assert parsed.human_review.authenticity_review_band is True
    assert parsed.human_review.notes == "Needs verification."
    assert parsed.rubric["leadership_through_action"] == 5
    assert parsed.rubric["growth_trajectory"] == 1
    assert parsed.rubric["community_orientation"] == 4
    assert parsed.rubric["project_based_readiness"] == 3
    assert parsed.rubric["motivation_groundedness"] == 2
    assert parsed.rubric["evidence_strength"] == 5
    assert parsed.rubric["shortlist_priority"] == 1
    assert parsed.evidence_bullets == ["Clear initiative."]


def test_llm_parser_keeps_backward_compatibility_for_old_explainability_shape() -> None:
    raw = """
    {
      "rubric_assessment": {
        "leadership_potential": 4,
        "growth_trajectory": 5,
        "motivation_authenticity": 4,
        "evidence_strength": 3,
        "hidden_potential_hint": 4,
        "authenticity_review_needed": "medium"
      },
      "top_strength_signals": [
        {
          "claim": "Shows initiative",
          "source": "motivation_questions",
          "snippet": "I organized a small study group and kept adjusting the format."
        }
      ],
      "main_gap_signals": [],
      "uncertainties": [
        {
          "claim": "Need stronger outcome evidence",
          "source": "interview_text",
          "snippet": "I can describe the work, but the impact numbers are limited."
        }
      ],
      "evidence_spans": [
        {
          "dimension": "initiative",
          "source": "motivation_questions",
          "text": "I organized a small study group and kept adjusting the format."
        }
      ],
      "committee_follow_up_question": "What changed after your first version failed?",
      "extractor_rationale": "Grounded in concrete actions and adaptation."
    }
    """

    parsed = parse_llm_extraction_json(raw)

    assert parsed.rubric_assessment["leadership_potential"] == 4
    assert parsed.rubric_assessment["growth_trajectory"] == 5
    assert parsed.rubric_assessment["authenticity_review_needed"] == "medium"
    assert parsed.committee_follow_up_question == "What changed after your first version failed?"
    assert parsed.uncertainty_signals[0].claim == "Need stronger outcome evidence"


def test_llm_parser_preserves_wrapper_answer_as_rationale() -> None:
    raw = """
    {
      "answer": "Candidate shows initiative and growth, but evidence is lightly quantified."
    }
    """

    parsed = parse_llm_extraction_json(raw)

    assert parsed.extractor_rationale == "Candidate shows initiative and growth, but evidence is lightly quantified."
