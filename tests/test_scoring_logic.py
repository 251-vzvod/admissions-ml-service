from copy import deepcopy
from types import SimpleNamespace

from app.config import CONFIG
from app.schemas.decision import Recommendation
from app.services.committee_guidance import build_committee_guidance
from app.services.llm_committee_writer import LLMCommitteeNarrativeResult
from app.services.llm_extractor import LLMAuthenticityAssist, LLMExplainabilityResult
from app.services.policy import build_policy_snapshot
from app.services.pipeline import ScoringPipeline


def test_generic_text_increases_risk() -> None:
    pipeline = ScoringPipeline()

    base_payload = {
        "candidate_id": "cand_logic_1",
        "structured_data": {
            "education": {
                "english_proficiency": {"type": "ielts", "score": 7.0},
                "school_certificate": {"type": "unt", "score": 105},
            }
        },
        "text_inputs": {
            "motivation_letter_text": "I organized a team project for 4 months and improved attendance by 25%.",
            "motivation_questions": [
                {
                    "question": "initiative",
                    "answer": "I created a student workshop and documented outcomes every week.",
                }
            ],
            "interview_text": "I can explain exactly what I did and what changed after the project.",
        },
    }

    generic_payload = deepcopy(base_payload)
    generic_payload["text_inputs"]["motivation_letter_text"] = (
        "I am passionate and motivated. I want to grow and change the world. " * 10
    )

    base_result = pipeline.score_candidate(base_payload)
    generic_result = pipeline.score_candidate(generic_payload)

    assert generic_result.authenticity_risk >= base_result.authenticity_risk


def test_remove_evidence_reduces_confidence() -> None:
    pipeline = ScoringPipeline()

    rich_payload = {
        "candidate_id": "cand_logic_2",
        "text_inputs": {
            "motivation_letter_text": "I led a volunteer project with 8 students and reduced waste by 30% in 2 months.",
            "motivation_questions": [
                {"question": "example", "answer": "I built a plan, assigned roles, and reported measurable outcomes."}
            ],
            "interview_text": "I can provide detailed steps and timeline of execution.",
        },
    }

    weak_payload = deepcopy(rich_payload)
    weak_payload["text_inputs"]["motivation_letter_text"] = "I am motivated and ready to grow."
    weak_payload["text_inputs"]["motivation_questions"] = [{"question": "example", "answer": "I am passionate."}]

    rich_result = pipeline.score_candidate(rich_payload)
    weak_result = pipeline.score_candidate(weak_payload)

    assert weak_result.confidence_score <= rich_result.confidence_score


def test_section_mismatch_increases_authenticity_risk() -> None:
    pipeline = ScoringPipeline()

    consistent_payload = {
        "candidate_id": "cand_consistent_sections",
        "text_inputs": {
            "motivation_letter_text": (
                "I started a peer study group for younger students, changed the format after the first month failed, "
                "and tracked what improved each week."
            ),
            "motivation_questions": [
                {
                    "question": "Tell us about your initiative.",
                    "answer": (
                        "I organized weekly sessions, collected feedback, and adjusted the plan when students were still confused."
                    ),
                }
            ],
            "interview_text": (
                "I can explain what failed at first, how I adapted the sessions, and what changed for the students afterward."
            ),
        },
    }

    inconsistent_payload = {
        "candidate_id": "cand_inconsistent_sections",
        "text_inputs": {
            "motivation_letter_text": (
                "I founded a coding club and built a long-term tutoring system for younger students."
            ),
            "motivation_questions": [
                {
                    "question": "Tell us about your initiative.",
                    "answer": "I mostly worked alone on environmental awareness posters and public speaking practice.",
                }
            ],
            "interview_text": (
                "My strongest experience is actually sports captaincy and event hosting, not tutoring or coding."
            ),
        },
    }

    consistent_result = pipeline.score_candidate(consistent_payload)
    inconsistent_result = pipeline.score_candidate(inconsistent_payload)

    assert inconsistent_result.authenticity_risk >= consistent_result.authenticity_risk


def test_llm_authenticity_assist_can_raise_risk_without_public_contract_change(monkeypatch) -> None:
    pipeline = ScoringPipeline()
    payload = {
        "candidate_id": "cand_llm_auth_assist",
        "text_inputs": {
            "motivation_letter_text": (
                "I want to create meaningful change and I believe this university will help me unlock my potential."
            ),
            "motivation_questions": [
                {
                    "question": "Tell us about your initiative.",
                    "answer": "I care about growth and impact but I did not describe one concrete project in detail.",
                }
            ],
            "interview_text": "I want to join a strong community and become a leader in the future.",
        },
    }

    old_enabled = CONFIG.llm.enabled
    try:
        CONFIG.llm.enabled = False
        baseline = pipeline.score_candidate(payload)

        CONFIG.llm.enabled = True
        monkeypatch.setattr(
            "app.services.pipeline.extract_explainability_with_llm",
            lambda **_: LLMExplainabilityResult(
                strength_claims=[],
                gap_claims=[
                    {
                        "claim": "Sections do not align well and strong claims remain under-supported.",
                        "source": "motivation_questions",
                        "snippet": "I did not describe one concrete project in detail.",
                    }
                ],
                uncertainty_claims=[
                    {
                        "claim": "Tone feels more polished than the amount of grounded evidence provided.",
                        "source": "motivation_letter_text",
                        "snippet": "unlock my potential",
                    }
                ],
                evidence_spans=[],
                rationale="Sections do not align strongly and the tone is more polished than grounded.",
                rubric_assessment={
                    "leadership_potential": 3,
                    "growth_trajectory": 3,
                    "motivation_authenticity": 2,
                    "evidence_strength": 2,
                    "hidden_potential_hint": 2,
                    "authenticity_review_needed": "high",
                },
                committee_follow_up_question="What is one concrete project you personally led from start to adjustment?",
                llm_metadata={"provider": "openai", "model": "gpt-4o-mini", "latency_ms": 1, "prompt_version": "test"},
                authenticity_assist=LLMAuthenticityAssist(
                    available=True,
                    review_needed="high",
                    risk_hint=0.84,
                    grounding_gap_score=0.78,
                    section_mismatch_score=0.71,
                    style_shift_score=0.63,
                    reasons=[
                        "Sections do not align well and strong claims remain under-supported.",
                        "Tone feels more polished than the amount of grounded evidence provided.",
                    ],
                ),
            ),
        )
        assisted = pipeline.score_candidate(payload)
    finally:
        CONFIG.llm.enabled = old_enabled

    assert assisted.authenticity_risk >= baseline.authenticity_risk
    assert "ai_detector" not in assisted.model_dump(mode="python")


def test_hidden_potential_is_not_punished_below_polished_low_signal_profile() -> None:
    pipeline = ScoringPipeline()

    hidden_potential_payload = {
        "candidate_id": "cand_hidden_potential",
        "text_inputs": {
            "motivation_letter_text": (
                "My father lost his job two years ago, so I started helping my younger brother study every evening. "
                "Later I organized shared notes for classmates before exams and kept improving them after feedback."
            ),
            "motivation_questions": [
                {
                    "question": "Tell us about your initiative.",
                    "answer": (
                        "Nobody asked me to do it. I made a simple study routine for three classmates, tracked what confused them, "
                        "and changed the plan each week when something failed."
                    ),
                }
            ],
            "interview_text": (
                "I am not the strongest speaker, but I can explain what I changed, why it helped, and what I learned from setbacks."
            ),
        },
    }

    polished_low_signal_payload = {
        "candidate_id": "cand_polished_low_signal",
        "text_inputs": {
            "motivation_letter_text": (
                "I am passionate, ambitious, and eager to grow into the best version of myself. "
                "I dream of changing the world through interdisciplinary teamwork and meaningful impact."
            ),
            "motivation_questions": [
                {
                    "question": "Tell us about your initiative.",
                    "answer": "I am deeply motivated to make a difference and bring positive energy to any environment.",
                }
            ],
            "interview_text": "I strongly believe this program will unlock my potential and shape my future.",
        },
    }

    hidden_result = pipeline.score_candidate(hidden_potential_payload)
    polished_result = pipeline.score_candidate(polished_low_signal_payload)

    assert hidden_result.merit_score >= polished_result.merit_score


def test_public_summary_uses_plain_language_not_pipeline_jargon() -> None:
    pipeline = ScoringPipeline()
    payload = {
        "candidate_id": "cand_summary_plain_language",
        "text_inputs": {
            "motivation_letter_text": (
                "I started a peer tutoring routine, changed it after attendance dropped, and kept helping younger students."
            ),
            "motivation_questions": [
                {
                    "question": "Why this university?",
                    "answer": "I want a place where I can keep building useful projects with other students.",
                }
            ],
            "interview_text": "I can explain what changed after the first version failed and what support I would still need.",
        },
    }

    result = pipeline.score_candidate(payload)

    summary = result.explanation.summary.lower()
    assert "deterministic feature extraction" not in summary
    assert "deterministic internal scoring" not in summary
    assert "this looks like" in summary or "this candidate" in summary or "there is not enough" in summary


def test_llm_committee_writer_can_override_public_narrative_fields_without_contract_change(monkeypatch) -> None:
    pipeline = ScoringPipeline()
    payload = {
        "candidate_id": "cand_committee_writer_override",
        "text_inputs": {
            "motivation_letter_text": (
                "I started a peer tutoring group, changed the format after attendance dropped, and kept helping younger students."
            ),
            "motivation_questions": [
                {
                    "question": "Why this university?",
                    "answer": "I want a place where I can keep building useful projects with other students.",
                }
            ],
            "interview_text": "I can explain what failed first and what changed after that.",
        },
    }

    old_enabled = CONFIG.llm.enabled
    try:
        CONFIG.llm.enabled = True
        monkeypatch.setattr(
            "app.services.pipeline.extract_explainability_with_llm",
            lambda **_: LLMExplainabilityResult(
                strength_claims=[],
                gap_claims=[],
                uncertainty_claims=[],
                evidence_spans=[],
                rationale="Evidence is grounded but still needs one verification step.",
                rubric_assessment={
                    "leadership_potential": 4,
                    "growth_trajectory": 4,
                    "motivation_authenticity": 4,
                    "evidence_strength": 3,
                    "hidden_potential_hint": 3,
                    "authenticity_review_needed": "low",
                },
                committee_follow_up_question="What changed after the first tutoring format failed?",
                llm_metadata={"provider": "openai", "model": "gpt-4o-mini", "latency_ms": 1, "prompt_version": "test"},
                authenticity_assist=LLMAuthenticityAssist(
                    available=True,
                    review_needed="low",
                    risk_hint=0.14,
                    grounding_gap_score=0.18,
                    section_mismatch_score=0.08,
                    style_shift_score=0.05,
                    reasons=[],
                ),
            ),
        )
        monkeypatch.setattr(
            "app.services.pipeline.generate_committee_narrative_with_llm",
            lambda **_: LLMCommitteeNarrativeResult(
                summary=(
                    "This candidate shows credible growth and practical initiative. "
                    "The main remaining question is how consistently that initiative translated into results for other students."
                ),
                committee_cohorts=["Trajectory-led candidate", "Community-oriented builder"],
                why_candidate_surfaced=[
                    "Shows grounded growth through a concrete tutoring example.",
                    "Deserves attention because the initiative is supported by adaptation rather than polish alone.",
                ],
                top_strengths=[
                    "Shows practical initiative backed by a concrete tutoring example.",
                    "Reflects on what failed and what changed afterward.",
                ],
                main_gaps=[
                    "The strongest claims would be easier to trust with one more concrete outcome.",
                ],
                uncertainties=[
                    "The exact scale of impact still needs one concrete measurable outcome.",
                ],
                what_to_verify_manually=[
                    "Ask for one measurable result from the tutoring group and the candidate's exact role in it.",
                ],
                suggested_follow_up_question="What changed in the tutoring group after you adjusted the first format?",
                llm_metadata={"provider": "openai", "model": "gpt-4o-mini", "latency_ms": 1, "writer_mode": "committee_narrative_v1"},
            ),
        )

        result = pipeline.score_candidate(payload)
    finally:
        CONFIG.llm.enabled = old_enabled

    public = result.model_dump(mode="python")
    assert result.explanation.summary.startswith("This candidate shows credible growth")
    assert result.committee_cohorts == ["Trajectory-led candidate", "Community-oriented builder"]
    assert result.why_candidate_surfaced[0].startswith("Shows grounded growth")
    assert result.top_strengths[0].startswith("Shows practical initiative")
    assert result.main_gaps[0].startswith("The strongest claims")
    assert result.uncertainties[0].startswith("The exact scale of impact")
    assert result.what_to_verify_manually[0].startswith("Ask for one measurable result")
    assert result.suggested_follow_up_question == "What changed in the tutoring group after you adjusted the first format?"
    assert "llm_metadata" not in public
    assert "supported_claims" not in public


def test_top_strengths_and_main_gaps_use_ui_friendly_evidence_format() -> None:
    pipeline = ScoringPipeline()
    payload = {
        "candidate_id": "cand_ui_friendly_strengths",
        "text_inputs": {
            "motivation_letter_text": (
                "I started a peer tutoring group, changed the format after attendance dropped, and kept helping younger students."
            ),
            "motivation_questions": [
                {
                    "question": "What changed after a mistake?",
                    "answer": "I learned to ask for feedback and adjust the plan instead of repeating the same mistake.",
                }
            ],
            "interview_text": "I can explain what failed first and what changed after that.",
        },
    }

    result = pipeline.score_candidate(payload)

    joined = " ".join([*result.top_strengths, *result.main_gaps]).lower()
    assert "[evidence:" not in joined
    assert "->" not in joined


def test_committee_guidance_surfaces_hidden_potential_and_follow_up_question() -> None:
    guidance = build_committee_guidance(
        review_signals={
            "growth_signal": 0.74,
            "agency_signal": 0.48,
            "motivation_signal": 0.58,
            "community_signal": 0.54,
            "evidence_signal": 0.57,
            "authenticity_signal": 0.79,
            "polish_risk_signal": 0.18,
            "hidden_signal": 0.64,
        },
        policy=build_policy_snapshot(
            merit_score=48,
            confidence_score=44,
            authenticity_risk=22,
            hidden_potential_score=38,
            support_needed_score=61,
            shortlist_priority_score=57,
            evidence_coverage_score=49,
            trajectory_score=52,
        ),
        hidden_potential_score=38,
        support_needed_score=61,
        trajectory_score=52,
        evidence_coverage_score=49,
        merit_score=48,
        authenticity_risk=22,
        recommendation=Recommendation.STANDARD_REVIEW,
        review_flags=[],
    )

    assert any("Hidden potential" == cohort for cohort in guidance.cohorts)
    assert any("Promising but needs support" == cohort for cohort in guidance.cohorts)
    assert any("Community-oriented builder" == cohort for cohort in guidance.cohorts)
    assert guidance.suggested_follow_up_question
    assert guidance.why_candidate_surfaced


def test_claim_evidence_extraction_returns_supported_or_weak_claims() -> None:
    pipeline = ScoringPipeline()

    payload = {
        "candidate_id": "cand_claim_map_001",
        "text_inputs": {
            "motivation_letter_text": (
                "I started a Saturday peer study group for younger students, changed the format after the first month failed, "
                "and tracked who returned each week."
            ),
            "motivation_questions": [
                {
                    "question": "What changed in you through this process?",
                    "answer": "I learned to collect feedback, adapt the plan, and keep responsibility when the first version did not work.",
                }
            ],
            "interview_text": "I can explain what failed first, what I changed, and what improved afterward.",
        },
    }

    result = pipeline.score_candidate(payload)

    assert result.supported_claims or result.weakly_supported_claims
    if result.supported_claims:
        first = result.supported_claims[0]
        assert first.claim
        assert first.source
        assert first.snippet
        assert 0 <= first.support_score <= 100


def test_llm_claims_can_enrich_evidence_highlights_without_contract_change(monkeypatch) -> None:
    pipeline = ScoringPipeline()
    payload = {
        "candidate_id": "cand_llm_claim_evidence",
        "text_inputs": {
            "motivation_letter_text": (
                "I started a student support notebook and updated it after classmates told me what was still confusing."
            ),
            "motivation_questions": [
                {
                    "question": "What changed after your first attempt?",
                    "answer": "I changed the plan after feedback and tracked which explanations helped more students return.",
                }
            ],
            "interview_text": "I can explain what I changed, why I changed it, and what improved after that.",
        },
    }

    old_enabled = CONFIG.llm.enabled
    try:
        CONFIG.llm.enabled = True
        monkeypatch.setattr(
            "app.services.pipeline.extract_explainability_with_llm",
            lambda **_: LLMExplainabilityResult(
                strength_claims=[
                    {
                        "claim": "Candidate adapts based on feedback and keeps responsibility after an initial weak result.",
                        "source": "interview_text",
                        "snippet": "I can explain what I changed, why I changed it, and what improved after that.",
                    }
                ],
                gap_claims=[],
                uncertainty_claims=[],
                evidence_spans=[
                    {
                        "dimension": "growth_trajectory",
                        "source": "interview_text",
                        "snippet": "I can explain what I changed, why I changed it, and what improved after that.",
                    }
                ],
                rationale="The strongest signal is adaptation grounded in one specific change process.",
                rubric_assessment={
                    "leadership_potential": 4,
                    "growth_trajectory": 4,
                    "motivation_authenticity": 3,
                    "evidence_strength": 3,
                    "hidden_potential_hint": 3,
                    "authenticity_review_needed": "low",
                },
                committee_follow_up_question="What specifically improved after you changed the support notebook format?",
                llm_metadata={"provider": "openai", "model": "gpt-4o-mini", "latency_ms": 1, "prompt_version": "test"},
                authenticity_assist=LLMAuthenticityAssist(
                    available=True,
                    review_needed="low",
                    risk_hint=0.18,
                    grounding_gap_score=0.22,
                    section_mismatch_score=0.10,
                    style_shift_score=0.06,
                    reasons=[],
                ),
            ),
        )
        monkeypatch.setattr(
            "app.services.pipeline.generate_committee_narrative_with_llm",
            lambda **_: LLMCommitteeNarrativeResult(
                summary="This candidate shows grounded adaptation after feedback.",
                committee_cohorts=["Trajectory-led candidate"],
                why_candidate_surfaced=["Shows one concrete example of adapting after feedback."],
                top_strengths=["Adapts based on feedback and explains what changed."],
                main_gaps=["The committee should still verify the exact outcome."],
                uncertainties=["The measurable outcome still needs one concrete example."],
                what_to_verify_manually=["Ask for the clearest result after the plan changed."],
                suggested_follow_up_question="What specifically improved after you changed the support notebook format?",
                llm_metadata={"provider": "openai", "model": "gpt-4o-mini", "latency_ms": 1, "writer_mode": "committee_narrative_v1"},
            ),
        )

        result = pipeline.score_candidate(payload)
    finally:
        CONFIG.llm.enabled = old_enabled

    assert result.evidence_highlights
    assert any("adapts based on feedback" in item.claim.lower() for item in result.evidence_highlights)


def test_community_oriented_candidate_outscores_self_advancement_only_motivation() -> None:
    pipeline = ScoringPipeline()

    community_payload = {
        "candidate_id": "cand_community_signal",
        "text_inputs": {
            "motivation_letter_text": (
                "In my city many younger students stop trying because they think extra learning is only for richer families. "
                "I started sharing my notes, helping them after class, and asking teachers what confused them most."
            ),
            "motivation_questions": [
                {
                    "question": "Why this program?",
                    "answer": (
                        "I want to learn how to build projects that solve real problems in my city and then bring that experience back."
                    ),
                }
            ],
            "interview_text": (
                "What matters to me is not only my future, but whether I can improve opportunities for other young people around me."
            ),
        },
    }

    self_advancement_payload = {
        "candidate_id": "cand_self_advancement_signal",
        "text_inputs": {
            "motivation_letter_text": (
                "I want to join this program because it will help me become more successful, more competitive, and more impressive."
            ),
            "motivation_questions": [
                {
                    "question": "Why this program?",
                    "answer": "I want stronger credentials, better networks, and a better personal future.",
                }
            ],
            "interview_text": "My main goal is to maximize my own opportunities and stand out.",
        },
    }

    community_result = pipeline.score_candidate(community_payload)
    self_advancement_result = pipeline.score_candidate(self_advancement_payload)

    assert community_result.merit_score >= self_advancement_result.merit_score
    assert community_result.semantic_rubric_scores["community_orientation"] >= self_advancement_result.semantic_rubric_scores["community_orientation"]


def test_academic_readiness_strengthens_merit_for_equal_text_signal() -> None:
    pipeline = ScoringPipeline()

    base_payload = {
        "candidate_id": "cand_academic_ready_base",
        "structured_data": {
            "education": {
                "english_proficiency": {"type": "ielts", "score": 6.0},
                "school_certificate": {"type": "unt", "score": 90},
            }
        },
        "text_inputs": {
            "motivation_letter_text": (
                "I organized a small peer study group, improved the format after feedback, and kept helping classmates prepare."
            ),
            "motivation_questions": [
                {
                    "question": "What did you change after feedback?",
                    "answer": "I changed the study plan after seeing what confused students most.",
                }
            ],
            "interview_text": "I can explain what changed, why I changed it, and what improved.",
        },
    }

    strong_academic_payload = deepcopy(base_payload)
    strong_academic_payload["candidate_id"] = "cand_academic_ready_strong"
    strong_academic_payload["structured_data"]["education"]["english_proficiency"] = {"type": "ielts", "score": 9.0}
    strong_academic_payload["structured_data"]["education"]["school_certificate"] = {"type": "unt", "score": 140}

    base_result = pipeline.score_candidate(base_payload)
    strong_result = pipeline.score_candidate(strong_academic_payload)

    assert strong_result.merit_score > base_result.merit_score


def test_academic_readiness_reduces_support_needed_for_equal_text_signal() -> None:
    pipeline = ScoringPipeline()

    base_payload = {
        "candidate_id": "cand_support_need_base",
        "structured_data": {
            "education": {
                "english_proficiency": {"type": "ielts", "score": 5.5},
                "school_certificate": {"type": "unt", "score": 85},
            }
        },
        "text_inputs": {
            "motivation_letter_text": (
                "I kept trying after the first version of my study plan failed and I asked for feedback each week."
            ),
            "motivation_questions": [
                {
                    "question": "What is difficult for you?",
                    "answer": "I still need support to adapt quickly, but I change my approach when I see a problem.",
                }
            ],
            "interview_text": "I can explain what support helps me perform better at the start.",
        },
    }

    strong_academic_payload = deepcopy(base_payload)
    strong_academic_payload["candidate_id"] = "cand_support_need_strong"
    strong_academic_payload["structured_data"]["education"]["english_proficiency"] = {"type": "ielts", "score": 8.5}
    strong_academic_payload["structured_data"]["education"]["school_certificate"] = {"type": "unt", "score": 135}

    base_result = pipeline.score_candidate(base_payload)
    strong_result = pipeline.score_candidate(strong_academic_payload)

    assert strong_result.support_needed_score < base_result.support_needed_score


def test_single_source_application_reduces_confidence_more_than_merit() -> None:
    pipeline = ScoringPipeline()

    core_story = (
        "I organized a peer study routine for younger students, changed the plan after feedback, "
        "and tracked what helped them return each week."
    )
    single_source_payload = {
        "candidate_id": "cand_single_source_confidence",
        "text_inputs": {
            "motivation_letter_text": core_story,
            "motivation_questions": [],
            "interview_text": "",
        },
    }

    multi_source_payload = {
        "candidate_id": "cand_multi_source_confidence",
        "text_inputs": {
            "motivation_letter_text": core_story,
            "motivation_questions": [
                {
                    "question": "What changed after feedback?",
                    "answer": "I changed the study plan after students told me which parts still confused them.",
                }
            ],
            "interview_text": "I can explain what failed first, what I changed, and what improved after that.",
        },
    }

    single_result = pipeline.score_candidate(single_source_payload)
    multi_result = pipeline.score_candidate(multi_source_payload)

    assert single_result.confidence_score < multi_result.confidence_score
    assert single_result.evidence_coverage_score <= multi_result.evidence_coverage_score
