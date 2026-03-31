"""Committee-facing guidance derived from transparent scoring signals."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.decision import Recommendation, ReviewFlag
from app.services.policy import RoutingPolicySnapshot


@dataclass(slots=True)
class CommitteeGuidance:
    cohorts: list[str]
    why_candidate_surfaced: list[str]
    what_to_verify_manually: list[str]
    suggested_follow_up_question: str


def build_committee_guidance(
    review_signals: dict[str, float],
    policy: RoutingPolicySnapshot,
    hidden_potential_score: int,
    support_needed_score: int,
    trajectory_score: int,
    evidence_coverage_score: int,
    merit_score: int,
    authenticity_risk: int,
    recommendation: Recommendation | str,
    review_flags: list[ReviewFlag | str],
) -> CommitteeGuidance:
    """Map transparent signals into committee-ready cohorts and follow-up guidance."""
    cohorts: list[str] = []
    why_candidate_surfaced: list[str] = []
    what_to_verify_manually: list[str] = []

    growth = float(review_signals.get("growth_signal", 0.0))
    agency = float(review_signals.get("agency_signal", 0.0))
    evidence = float(review_signals.get("evidence_signal", 0.0))
    motivation = float(review_signals.get("motivation_signal", 0.0))
    community = float(review_signals.get("community_signal", 0.0))
    polish_risk = float(review_signals.get("polish_risk_signal", 0.0))
    authenticity = float(review_signals.get("authenticity_signal", 0.0))
    hidden = float(review_signals.get("hidden_signal", 0.0))

    hidden_candidate = policy.hidden_potential_band
    support_needed = policy.support_needed_band
    polished_low_evidence = (
        polish_risk >= 0.42
        and evidence_coverage_score < 44
        and trajectory_score < 24
        and not policy.hidden_potential_band
    )
    authenticity_review = policy.authenticity_review_band or any(
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

    if policy.priority_band or recommendation == Recommendation.REVIEW_PRIORITY or merit_score >= 55:
        cohorts.append("High priority")
    if hidden_candidate:
        cohorts.append("Hidden potential")
    if trajectory_score >= 18 or (growth >= 0.28 and evidence >= 0.22):
        cohorts.append("Trajectory-led candidate")
    if community >= 0.46:
        cohorts.append("Community-oriented builder")
    if support_needed:
        cohorts.append("Promising but needs support")
    if polished_low_evidence:
        cohorts.append("Polished but low-evidence")
    if authenticity_review:
        cohorts.append("Authenticity review needed")
    if not cohorts:
        cohorts.append("Standard committee review")

    if trajectory_score >= 18 or (growth >= 0.28 and evidence >= 0.22):
        why_candidate_surfaced.append("Strong growth trajectory and reflection signals.")
    if agency >= 0.54:
        why_candidate_surfaced.append("Clear agency or leadership markers in actions described.")
    if evidence >= 0.60:
        why_candidate_surfaced.append("Application includes enough concrete evidence to support prioritization.")
    if hidden_candidate:
        why_candidate_surfaced.append("Underlying signal appears stronger than the candidate's self-presentation.")
    if hidden_candidate and evidence < 0.58:
        why_candidate_surfaced.append("Candidate shows early-stage leadership potential even with imperfect self-presentation.")
    if community >= 0.46:
        why_candidate_surfaced.append("Motivation is tied to usefulness, responsibility, or improving life for other people.")
    if motivation >= 0.62 and not polished_low_evidence:
        why_candidate_surfaced.append("Motivation appears specific enough to justify committee attention.")
    why_candidate_surfaced = why_candidate_surfaced[:3]

    if evidence < 0.45:
        what_to_verify_manually.append("Ask for one concrete example with actions, obstacles, and measurable outcome.")
    if authenticity_review or authenticity < 0.48:
        what_to_verify_manually.append("Verify that key claims stay consistent across essay, Q/A, and interview-style answers.")
    if support_needed:
        what_to_verify_manually.append("Check what language, academic, or onboarding support would help this candidate convert potential into performance.")
    if hidden_candidate and not authenticity_review:
        what_to_verify_manually.append("Probe for additional leadership evidence that may not be visible from writing quality alone.")
    if community >= 0.46:
        what_to_verify_manually.append("Ask for one example where the candidate improved something for a real group, not only for themselves.")
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
    elif community >= 0.46:
        suggested_follow_up_question = (
            "Tell us about one real problem around you that you tried to improve for other people. What exactly did you do, and what changed?"
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
