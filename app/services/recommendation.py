"""Workflow recommendation mapping.

Recommendation is a committee routing signal and not a final admission decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import CONFIG


@dataclass(slots=True)
class RecommendationResult:
    recommendation: str
    review_flags: list[str]


def map_recommendation(
    eligibility_status: str,
    merit_raw: float,
    confidence_raw: float,
    authenticity_risk_raw: float,
    feature_map: dict[str, float | bool],
    prior_flags: list[str],
) -> RecommendationResult:
    """Map scores + eligibility into workflow recommendation categories."""
    flags = list(prior_flags)

    if eligibility_status == "invalid":
        return RecommendationResult(recommendation="invalid", review_flags=flags)

    if eligibility_status == "incomplete_application":
        return RecommendationResult(recommendation="incomplete_application", review_flags=flags)

    evidence_count = float(feature_map.get("evidence_count", 0.0))

    if confidence_raw < CONFIG.thresholds.very_low_confidence and evidence_count < CONFIG.thresholds.low_evidence:
        if "insufficient_evidence" not in flags:
            flags.append("insufficient_evidence")
        return RecommendationResult(recommendation="insufficient_evidence", review_flags=flags)

    if merit_raw >= CONFIG.thresholds.high_merit and confidence_raw >= CONFIG.thresholds.acceptable_confidence and authenticity_risk_raw < 0.45:
        return RecommendationResult(recommendation="review_priority", review_flags=flags)

    if merit_raw >= CONFIG.thresholds.medium_merit:
        if confidence_raw < CONFIG.thresholds.acceptable_confidence or authenticity_risk_raw >= CONFIG.thresholds.elevated_risk:
            return RecommendationResult(recommendation="manual_review_required", review_flags=flags)
        return RecommendationResult(recommendation="standard_review", review_flags=flags)

    if authenticity_risk_raw >= CONFIG.thresholds.high_risk:
        return RecommendationResult(recommendation="manual_review_required", review_flags=flags)

    return RecommendationResult(recommendation="standard_review", review_flags=flags)
