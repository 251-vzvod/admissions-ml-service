"""Workflow recommendation mapping.

Recommendation is a committee routing signal and not a final admission decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import CONFIG
from app.schemas.decision import Recommendation, ReviewFlag


@dataclass(slots=True)
class RecommendationResult:
    recommendation: Recommendation
    review_flags: list[ReviewFlag]


def map_recommendation(
    eligibility_status: str,
    merit_raw: float,
    confidence_raw: float,
    authenticity_risk_raw: float,
    feature_map: dict[str, float | bool],
    prior_flags: list[ReviewFlag],
) -> RecommendationResult:
    """Map scores + eligibility into workflow recommendation categories."""
    flags = list(prior_flags)

    if confidence_raw < CONFIG.thresholds.acceptable_confidence:
        flags.append(ReviewFlag.LOW_CONFIDENCE)

    if eligibility_status == Recommendation.INVALID:
        return RecommendationResult(recommendation=Recommendation.INVALID, review_flags=flags)

    if eligibility_status == Recommendation.INCOMPLETE_APPLICATION:
        return RecommendationResult(recommendation=Recommendation.INCOMPLETE_APPLICATION, review_flags=flags)

    evidence_count = float(feature_map.get("evidence_count", 0.0))

    if confidence_raw < CONFIG.thresholds.very_low_confidence and evidence_count < CONFIG.thresholds.low_evidence:
        if ReviewFlag.INSUFFICIENT_EVIDENCE not in flags:
            flags.append(ReviewFlag.INSUFFICIENT_EVIDENCE)
        return RecommendationResult(recommendation=Recommendation.INSUFFICIENT_EVIDENCE, review_flags=flags)

    if merit_raw >= CONFIG.thresholds.high_merit and confidence_raw >= CONFIG.thresholds.acceptable_confidence and authenticity_risk_raw < 0.45:
        return RecommendationResult(recommendation=Recommendation.REVIEW_PRIORITY, review_flags=flags)

    if merit_raw >= CONFIG.thresholds.medium_merit:
        if confidence_raw < CONFIG.thresholds.acceptable_confidence or authenticity_risk_raw >= CONFIG.thresholds.elevated_risk:
            return RecommendationResult(recommendation=Recommendation.MANUAL_REVIEW_REQUIRED, review_flags=flags)
        return RecommendationResult(recommendation=Recommendation.STANDARD_REVIEW, review_flags=flags)

    if authenticity_risk_raw >= CONFIG.thresholds.high_risk:
        return RecommendationResult(recommendation=Recommendation.MANUAL_REVIEW_REQUIRED, review_flags=flags)

    return RecommendationResult(recommendation=Recommendation.STANDARD_REVIEW, review_flags=flags)
