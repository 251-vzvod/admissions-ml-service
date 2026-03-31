from copy import deepcopy

from app.schemas.decision import Recommendation
from app.services.committee_guidance import build_committee_guidance
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
