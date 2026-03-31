"""Committee-facing guidance derived from transparent scoring signals."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.decision import Recommendation, ReviewFlag


@dataclass(slots=True)
class CommitteeGuidance:
    cohorts: list[str]
    why_candidate_surfaced: list[str]
    what_to_verify_manually: list[str]
    suggested_follow_up_question: str


def build_committee_guidance(
    feature_map: dict[str, float | bool],
    semantic_scores: dict[str, float],
    hidden_potential_score: int,
    support_needed_score: int,
    trajectory_score: int,
    evidence_coverage_score: int,
    merit_score: int,
    confidence_score: int,
    authenticity_risk: int,
    recommendation: Recommendation | str,
    review_flags: list[ReviewFlag | str],
) -> CommitteeGuidance:
    """Map transparent signals into committee-ready cohorts and follow-up guidance."""
    cohorts: list[str] = []
    why_candidate_surfaced: list[str] = []
    what_to_verify_manually: list[str] = []

    growth = float(feature_map.get("growth_trajectory", 0.0))
    resilience = float(feature_map.get("resilience", 0.0))
    initiative = float(feature_map.get("initiative", 0.0))
    evidence_count = float(feature_map.get("evidence_count", 0.0))
    specificity = float(feature_map.get("specificity_score", 0.0))
    motivation = float(feature_map.get("motivation_clarity", 0.0))
    genericness = float(feature_map.get("genericness_score", 0.0))
    polished_empty = float(feature_map.get("polished_but_empty_score", 0.0))
    semantic_hidden = float(semantic_scores.get("hidden_potential", 0.0))
    semantic_leadership = float(semantic_scores.get("leadership_potential", 0.0))
    semantic_growth = float(semantic_scores.get("growth_trajectory", 0.0))

    hidden_candidate = (
        hidden_potential_score >= 22
        and evidence_coverage_score >= 20
        and trajectory_score >= 15
        and (growth >= 0.12 or semantic_hidden >= 0.48)
    )
    support_needed = support_needed_score >= 45
    polished_low_evidence = polished_empty >= 0.35 and evidence_coverage_score < 42
    authenticity_review = authenticity_risk >= 45 or any(
        flag in review_flags
        for flag in {
            ReviewFlag.HIGH_AUTHENTICITY_RISK,
            ReviewFlag.MODERATE_AUTHENTICITY_RISK,
            ReviewFlag.CROSS_SECTION_MISMATCH,
            ReviewFlag.SECTION_MISMATCH,
            ReviewFlag.POSSIBLE_CONTRADICTION,
            ReviewFlag.CONTRADICTION_RISK,
        }
    )

    if recommendation == Recommendation.REVIEW_PRIORITY or merit_score >= 55:
        cohorts.append("High priority")
    if hidden_candidate:
        cohorts.append("Hidden potential")
    if trajectory_score >= 18 or (growth >= 0.18 and evidence_count >= 0.10):
        cohorts.append("Trajectory-led candidate")
    if support_needed:
        cohorts.append("Promising but needs support")
    if polished_low_evidence:
        cohorts.append("Polished but low-evidence")
    if authenticity_review:
        cohorts.append("Authenticity review needed")
    if not cohorts:
        cohorts.append("Standard committee review")

    if trajectory_score >= 18 or (growth >= 0.18 and evidence_count >= 0.10):
        why_candidate_surfaced.append("Strong growth trajectory and reflection signals.")
    if initiative >= 0.60 or semantic_leadership >= 0.60:
        why_candidate_surfaced.append("Clear agency or leadership markers in actions described.")
    if evidence_count >= 0.65:
        why_candidate_surfaced.append("Application includes enough concrete evidence to support prioritization.")
    if hidden_candidate:
        why_candidate_surfaced.append("Underlying signal appears stronger than the candidate's self-presentation.")
    if hidden_candidate and specificity < 0.58:
        why_candidate_surfaced.append("Candidate shows early-stage leadership potential even with imperfect self-presentation.")
    if motivation >= 0.70 and not polished_low_evidence:
        why_candidate_surfaced.append("Motivation appears specific enough to justify committee attention.")
    why_candidate_surfaced = why_candidate_surfaced[:3]

    if specificity < 0.45 or evidence_count < 0.40:
        what_to_verify_manually.append("Ask for one concrete example with actions, obstacles, and measurable outcome.")
    if authenticity_review:
        what_to_verify_manually.append("Verify that key claims stay consistent across essay, Q/A, and interview-style answers.")
    if support_needed:
        what_to_verify_manually.append("Check what language, academic, or onboarding support would help this candidate convert potential into performance.")
    if hidden_candidate and not authenticity_review:
        what_to_verify_manually.append("Probe for additional leadership evidence that may not be visible from writing quality alone.")
    what_to_verify_manually = what_to_verify_manually[:3]

    if authenticity_review:
        suggested_follow_up_question = (
            "Can you walk us through one specific example from your application step by step: what happened, what you personally did, and what changed as a result?"
        )
    elif hidden_candidate:
        suggested_follow_up_question = (
            "Tell us about a situation where you noticed a problem before others did and decided to act. What exactly did you do, and what changed afterward?"
        )
    elif support_needed:
        suggested_follow_up_question = (
            "Which part of the program do you think would challenge you most at the start, and how would you use support or feedback to adapt quickly?"
        )
    elif polished_low_evidence:
        suggested_follow_up_question = (
            "What is the strongest concrete result you have personally achieved so far, and how can you verify your role in it?"
        )
    else:
        suggested_follow_up_question = (
            "What is one example from your application that best shows how you create value for other people, not just for yourself?"
        )

    return CommitteeGuidance(
        cohorts=cohorts,
        why_candidate_surfaced=why_candidate_surfaced,
        what_to_verify_manually=what_to_verify_manually,
        suggested_follow_up_question=suggested_follow_up_question,
    )
