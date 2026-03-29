"""Scoring engine for merit, confidence, and breakdown axes."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import CONFIG
from app.utils.math_utils import clamp01, to_display_score, weighted_average_normalized


@dataclass(slots=True)
class ScoringResult:
    merit_raw: float
    confidence_raw: float
    authenticity_risk_raw: float
    merit_breakdown_raw: dict[str, float]
    merit_score: int
    confidence_score: int
    authenticity_risk: int


def compute_scores(feature_map: dict[str, float | bool], authenticity_risk_raw: float) -> ScoringResult:
    """Compute candidate-level operational scores for decision support."""
    f = feature_map

    potential = weighted_average_normalized(
        [
            (float(f.get("growth_trajectory", 0.0)), 0.30),
            (float(f.get("resilience", 0.0)), 0.20),
            (float(f.get("initiative", 0.0)), 0.25),
            (float(f.get("program_fit", 0.0)), 0.20),
            (float(f.get("english_score_normalized", 0.5)), 0.05),
        ]
    )
    motivation = weighted_average_normalized(
        [
            (float(f.get("motivation_clarity", 0.0)), 0.45),
            (float(f.get("program_fit", 0.0)), 0.30),
            (float(f.get("evidence_richness", 0.0)), 0.25),
        ]
    )
    leadership_agency = weighted_average_normalized(
        [
            (float(f.get("initiative", 0.0)), 0.35),
            (float(f.get("leadership_impact", 0.0)), 0.35),
            (float(f.get("evidence_richness", 0.0)), 0.15),
            (float(f.get("evidence_count", 0.0)), 0.15),
        ]
    )
    experience_skills = weighted_average_normalized(
        [
            (float(f.get("evidence_count", 0.0)), 0.30),
            (float(f.get("leadership_impact", 0.0)), 0.20),
            (float(f.get("achievement_mentions_count", 0.0)), 0.20),
            (float(f.get("project_mentions_count", 0.0)), 0.15),
            (float(f.get("english_score_normalized", 0.5)), 0.075),
            (float(f.get("certificate_score_normalized", 0.5)), 0.075),
        ]
    )

    trust_base = weighted_average_normalized(
        [
            (float(f.get("completeness_score", 0.0)), 0.30),
            (float(f.get("consistency_score", 0.0)), 0.25),
            (float(f.get("evidence_count", 0.0)), 0.20),
            (float(f.get("behavioral_completion_score", 0.0)), 0.10),
            (float(f.get("docs_count_score", 0.0)), 0.07),
            (float(f.get("portfolio_links_score", 0.0)), 0.05),
            (1.0 if bool(f.get("has_video_presentation", False)) else 0.0, 0.03),
        ]
    )
    trust_penalty = 0.0
    if bool(f.get("contradiction_flag", False)):
        trust_penalty += 0.15
    if bool(f.get("low_evidence_flag", False)):
        trust_penalty += 0.10
    trust_penalty += float(f.get("genericness_score", 0.0)) * 0.10
    trust_completeness = clamp01(trust_base - trust_penalty)

    merit_breakdown_raw = {
        "potential": potential,
        "motivation": motivation,
        "leadership_agency": leadership_agency,
        "experience_skills": experience_skills,
        "trust_completeness": trust_completeness,
    }

    merit_raw = weighted_average_normalized(
        [(value, CONFIG.weights.merit_breakdown[key]) for key, value in merit_breakdown_raw.items()]
    )

    confidence_base = weighted_average_normalized(
        [
            (float(f.get("specificity_score", 0.0)), CONFIG.weights.confidence_components["specificity_score"]),
            (float(f.get("evidence_count", 0.0)), CONFIG.weights.confidence_components["evidence_count"]),
            (float(f.get("consistency_score", 0.0)), CONFIG.weights.confidence_components["consistency_score"]),
            (float(f.get("completeness_score", 0.0)), CONFIG.weights.confidence_components["completeness_score"]),
        ]
    )

    confidence_penalty = 0.0
    if bool(f.get("contradiction_flag", False)):
        confidence_penalty += 0.15
    if bool(f.get("low_evidence_flag", False)):
        confidence_penalty += 0.12
    confidence_penalty += max(0.0, authenticity_risk_raw - 0.5) * 0.20

    confidence_raw = clamp01(confidence_base - confidence_penalty)

    return ScoringResult(
        merit_raw=merit_raw,
        confidence_raw=confidence_raw,
        authenticity_risk_raw=authenticity_risk_raw,
        merit_breakdown_raw=merit_breakdown_raw,
        merit_score=to_display_score(merit_raw),
        confidence_score=to_display_score(confidence_raw),
        authenticity_risk=to_display_score(authenticity_risk_raw),
    )
