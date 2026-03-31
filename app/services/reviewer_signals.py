"""Compact reviewer-facing signal layer built from transparent raw features."""

from __future__ import annotations

from app.utils.math_utils import clamp01, weighted_average_normalized


def build_reviewer_signals(feature_map: dict[str, float | bool | int | None]) -> dict[str, float]:
    """Aggregate raw features into a small set of reviewer-facing high-level signals."""
    growth_signal = weighted_average_normalized(
        [
            (float(feature_map.get("growth_trajectory", 0.0)), 0.34),
            (float(feature_map.get("resilience", 0.0)), 0.18),
            (float(feature_map.get("trajectory_adaptation_score", 0.0)), 0.18),
            (float(feature_map.get("trajectory_reflection_score", 0.0)), 0.14),
            (float(feature_map.get("semantic_growth_trajectory", 0.0)), 0.16),
        ]
    )
    agency_signal = weighted_average_normalized(
        [
            (float(feature_map.get("initiative", 0.0)), 0.34),
            (float(feature_map.get("leadership_impact", 0.0)), 0.24),
            (float(feature_map.get("trajectory_outcome_score", 0.0)), 0.10),
            (float(feature_map.get("evidence_count", 0.0)), 0.08),
            (float(feature_map.get("semantic_leadership_potential", 0.0)), 0.24),
        ]
    )
    motivation_signal = weighted_average_normalized(
        [
            (float(feature_map.get("motivation_clarity", 0.0)), 0.32),
            (float(feature_map.get("program_fit", 0.0)), 0.24),
            (float(feature_map.get("specificity_score", 0.0)), 0.14),
            (float(feature_map.get("evidence_richness", 0.0)), 0.10),
            (float(feature_map.get("semantic_motivation_authenticity", 0.0)), 0.20),
        ]
    )
    community_signal = weighted_average_normalized(
        [
            (float(feature_map.get("community_value_orientation", 0.0)), 0.62),
            (float(feature_map.get("semantic_community_orientation", 0.0)), 0.38),
        ]
    )
    evidence_signal = weighted_average_normalized(
        [
            (float(feature_map.get("evidence_count", 0.0)), 0.28),
            (float(feature_map.get("specificity_score", 0.0)), 0.24),
            (float(feature_map.get("evidence_richness", 0.0)), 0.18),
            (float(feature_map.get("consistency_score", 0.0)), 0.18),
            (float(feature_map.get("completeness_score", 0.0)), 0.12),
        ]
    )
    authenticity_signal = weighted_average_normalized(
        [
            (float(feature_map.get("consistency_score", 0.0)), 0.24),
            (1.0 - float(feature_map.get("cross_section_mismatch_score", 0.0)), 0.22),
            (1.0 - float(feature_map.get("genericness_score", 0.0)), 0.14),
            (1.0 - float(feature_map.get("polished_but_empty_score", 0.0)), 0.14),
            (float(feature_map.get("semantic_authenticity_groundedness", 0.0)), 0.18),
            (1.0 - float(feature_map.get("authenticity_risk_raw", 0.0)), 0.08),
        ]
    )
    polish_risk_signal = weighted_average_normalized(
        [
            (float(feature_map.get("polished_but_empty_score", 0.0)), 0.34),
            (float(feature_map.get("genericness_score", 0.0)), 0.26),
            (float(feature_map.get("cross_section_mismatch_score", 0.0)), 0.18),
            (1.0 - float(feature_map.get("specificity_score", 0.0)), 0.12),
            (1.0 - float(feature_map.get("evidence_count", 0.0)), 0.10),
        ]
    )
    hidden_signal = weighted_average_normalized(
        [
            (float(feature_map.get("semantic_hidden_potential", 0.0)), 0.42),
            (growth_signal, 0.24),
            (agency_signal, 0.16),
            (evidence_signal, 0.10),
            (clamp01(1.0 - polish_risk_signal), 0.08),
        ]
    )
    return {
        "growth_signal": round(growth_signal, 4),
        "agency_signal": round(agency_signal, 4),
        "motivation_signal": round(motivation_signal, 4),
        "community_signal": round(community_signal, 4),
        "evidence_signal": round(evidence_signal, 4),
        "authenticity_signal": round(authenticity_signal, 4),
        "polish_risk_signal": round(polish_risk_signal, 4),
        "hidden_signal": round(hidden_signal, 4),
    }
