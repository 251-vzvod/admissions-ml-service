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


def _component_rows(items: list[tuple[str, float, float]]) -> tuple[list[dict[str, float | str]], float]:
    rows: list[dict[str, float | str]] = []
    total_weight = sum(weight for _, _, weight in items)
    weighted_sum = 0.0
    for feature, value, weight in items:
        contribution = value * weight
        weighted_sum += contribution
        rows.append(
            {
                "feature": feature,
                "value": round(value, 6),
                "weight": round(weight, 6),
                "contribution": round(contribution, 6),
            }
        )
    raw = weighted_sum / total_weight if total_weight else 0.0
    return rows, raw


def build_score_trace(
    feature_map: dict[str, float | bool],
    authenticity_risk_raw: float,
    use_semantic_layer: bool = True,
) -> dict[str, object]:
    """Build auditable factor-level trace for deterministic scoring formulas."""
    f = feature_map

    potential_items = [
        ("growth_trajectory", float(f.get("growth_trajectory", 0.0)), 0.32),
        ("resilience", float(f.get("resilience", 0.0)), 0.23),
        ("initiative", float(f.get("initiative", 0.0)), 0.25),
        ("program_fit", float(f.get("program_fit", 0.0)), 0.10),
        ("evidence_richness", float(f.get("evidence_richness", 0.0)), 0.10),
    ]
    if use_semantic_layer:
        potential_items.extend(
            [
                ("semantic_growth_trajectory", float(f.get("semantic_growth_trajectory", 0.0)), 0.18),
                ("semantic_leadership_potential", float(f.get("semantic_leadership_potential", 0.0)), 0.12),
            ]
        )
    potential_rows, potential_raw = _component_rows(potential_items)

    motivation_items = [
        ("motivation_clarity", float(f.get("motivation_clarity", 0.0)), 0.45),
        ("program_fit", float(f.get("program_fit", 0.0)), 0.30),
        ("evidence_richness", float(f.get("evidence_richness", 0.0)), 0.25),
    ]
    if use_semantic_layer:
        motivation_items.append(("semantic_motivation_authenticity", float(f.get("semantic_motivation_authenticity", 0.0)), 0.30))
    motivation_rows, motivation_raw = _component_rows(motivation_items)

    leadership_items = [
        ("initiative", float(f.get("initiative", 0.0)), 0.35),
        ("leadership_impact", float(f.get("leadership_impact", 0.0)), 0.35),
        ("evidence_richness", float(f.get("evidence_richness", 0.0)), 0.15),
        ("evidence_count", float(f.get("evidence_count", 0.0)), 0.15),
    ]
    if use_semantic_layer:
        leadership_items.extend(
            [
                ("semantic_leadership_potential", float(f.get("semantic_leadership_potential", 0.0)), 0.30),
                ("semantic_hidden_potential", float(f.get("semantic_hidden_potential", 0.0)), 0.10),
            ]
        )
    leadership_rows, leadership_raw = _component_rows(leadership_items)

    experience_items = [
        ("evidence_count", float(f.get("evidence_count", 0.0)), 0.35),
        ("leadership_impact", float(f.get("leadership_impact", 0.0)), 0.25),
        ("achievement_mentions_count", float(f.get("achievement_mentions_count", 0.0)), 0.20),
        ("project_mentions_count", float(f.get("project_mentions_count", 0.0)), 0.20),
    ]
    experience_rows, experience_raw = _component_rows(experience_items)

    trust_items = [
        ("completeness_score", float(f.get("completeness_score", 0.0)), 0.36),
        ("consistency_score", float(f.get("consistency_score", 0.0)), 0.30),
        ("evidence_count", float(f.get("evidence_count", 0.0)), 0.22),
        ("behavioral_completion_score", float(f.get("behavioral_completion_score", 0.0)), 0.12),
    ]
    if use_semantic_layer:
        trust_items.append(("semantic_authenticity_groundedness", float(f.get("semantic_authenticity_groundedness", 0.0)), 0.18))
    trust_rows, trust_base = _component_rows(trust_items)

    trust_penalty = 0.0
    trust_penalty_contrib = {
        "contradiction_flag": 0.15 if bool(f.get("contradiction_flag", False)) else 0.0,
        "low_evidence_flag": 0.10 if bool(f.get("low_evidence_flag", False)) else 0.0,
        "genericness_penalty": float(f.get("genericness_score", 0.0)) * 0.10,
    }
    trust_penalty += trust_penalty_contrib["contradiction_flag"]
    trust_penalty += trust_penalty_contrib["low_evidence_flag"]
    trust_penalty += trust_penalty_contrib["genericness_penalty"]
    trust_raw = clamp01(trust_base - trust_penalty)

    merit_breakdown = {
        "potential": potential_raw,
        "motivation": motivation_raw,
        "leadership_agency": leadership_raw,
        "experience_skills": experience_raw,
        "trust_completeness": trust_raw,
    }

    merit_components = [
        (axis, merit_breakdown[axis], CONFIG.weights.merit_breakdown[axis])
        for axis in ["potential", "motivation", "leadership_agency", "experience_skills", "trust_completeness"]
    ]
    merit_rows, merit_raw = _component_rows(merit_components)

    confidence_items = [
        ("specificity_score", float(f.get("specificity_score", 0.0)), CONFIG.weights.confidence_components["specificity_score"]),
        ("evidence_count", float(f.get("evidence_count", 0.0)), CONFIG.weights.confidence_components["evidence_count"]),
        ("consistency_score", float(f.get("consistency_score", 0.0)), CONFIG.weights.confidence_components["consistency_score"]),
        ("completeness_score", float(f.get("completeness_score", 0.0)), CONFIG.weights.confidence_components["completeness_score"]),
    ]
    if use_semantic_layer:
        confidence_items.append(("semantic_authenticity_groundedness", float(f.get("semantic_authenticity_groundedness", 0.0)), 0.22))
    confidence_rows, confidence_base = _component_rows(confidence_items)

    confidence_penalty = 0.0
    confidence_penalty_contrib = {
        "contradiction_flag": 0.15 if bool(f.get("contradiction_flag", False)) else 0.0,
        "low_evidence_flag": 0.12 if bool(f.get("low_evidence_flag", False)) else 0.0,
        "authenticity_risk_penalty": max(0.0, authenticity_risk_raw - 0.5) * 0.20,
    }
    confidence_penalty += confidence_penalty_contrib["contradiction_flag"]
    confidence_penalty += confidence_penalty_contrib["low_evidence_flag"]
    confidence_penalty += confidence_penalty_contrib["authenticity_risk_penalty"]
    confidence_raw = clamp01(confidence_base - confidence_penalty)

    return {
        "formulas": {
            "potential": "weighted_average([growth_trajectory,resilience,initiative,program_fit,evidence_richness,semantic_growth_trajectory?,semantic_leadership_potential?])",
            "motivation": "weighted_average([motivation_clarity,program_fit,evidence_richness,semantic_motivation_authenticity?])",
            "leadership_agency": "weighted_average([initiative,leadership_impact,evidence_richness,evidence_count,semantic_leadership_potential?,semantic_hidden_potential?])",
            "experience_skills": "weighted_average([evidence_count,leadership_impact,achievement_mentions_count,project_mentions_count])",
            "trust_completeness": "clamp01(weighted_average(trust_features) - trust_penalty)",
            "merit": "weighted_average([potential,motivation,leadership_agency,experience_skills,trust_completeness], merit_weights)",
            "confidence": "clamp01(weighted_average(confidence_features) - confidence_penalty)",
        },
        "components": {
            "potential": potential_rows,
            "motivation": motivation_rows,
            "leadership_agency": leadership_rows,
            "experience_skills": experience_rows,
            "trust_completeness_base": trust_rows,
            "merit": merit_rows,
            "confidence_base": confidence_rows,
        },
        "penalties": {
            "trust_penalty": {
                "components": {k: round(v, 6) for k, v in trust_penalty_contrib.items()},
                "total": round(trust_penalty, 6),
            },
            "confidence_penalty": {
                "components": {k: round(v, 6) for k, v in confidence_penalty_contrib.items()},
                "total": round(confidence_penalty, 6),
            },
        },
        "axis_raw": {k: round(v, 6) for k, v in merit_breakdown.items()},
        "outputs": {
            "merit_raw": round(merit_raw, 6),
            "confidence_raw": round(confidence_raw, 6),
            "authenticity_risk_raw": round(authenticity_risk_raw, 6),
            "merit_score": to_display_score(merit_raw),
            "confidence_score": to_display_score(confidence_raw),
            "authenticity_risk": to_display_score(authenticity_risk_raw),
        },
    }


def compute_scores(
    feature_map: dict[str, float | bool],
    authenticity_risk_raw: float,
    use_semantic_layer: bool = True,
) -> ScoringResult:
    """Compute candidate-level operational scores for decision support."""
    f = feature_map

    potential_items = [
        (float(f.get("growth_trajectory", 0.0)), 0.32),
        (float(f.get("resilience", 0.0)), 0.23),
        (float(f.get("initiative", 0.0)), 0.25),
        (float(f.get("program_fit", 0.0)), 0.10),
        (float(f.get("evidence_richness", 0.0)), 0.10),
    ]
    if use_semantic_layer:
        potential_items.extend(
            [
                (float(f.get("semantic_growth_trajectory", 0.0)), 0.18),
                (float(f.get("semantic_leadership_potential", 0.0)), 0.12),
            ]
        )
    potential = weighted_average_normalized(potential_items)

    motivation_items = [
        (float(f.get("motivation_clarity", 0.0)), 0.45),
        (float(f.get("program_fit", 0.0)), 0.30),
        (float(f.get("evidence_richness", 0.0)), 0.25),
    ]
    if use_semantic_layer:
        motivation_items.append((float(f.get("semantic_motivation_authenticity", 0.0)), 0.30))
    motivation = weighted_average_normalized(motivation_items)

    leadership_items = [
        (float(f.get("initiative", 0.0)), 0.35),
        (float(f.get("leadership_impact", 0.0)), 0.35),
        (float(f.get("evidence_richness", 0.0)), 0.15),
        (float(f.get("evidence_count", 0.0)), 0.15),
    ]
    if use_semantic_layer:
        leadership_items.extend(
            [
                (float(f.get("semantic_leadership_potential", 0.0)), 0.30),
                (float(f.get("semantic_hidden_potential", 0.0)), 0.10),
            ]
        )
    leadership_agency = weighted_average_normalized(leadership_items)
    experience_skills = weighted_average_normalized(
        [
            (float(f.get("evidence_count", 0.0)), 0.35),
            (float(f.get("leadership_impact", 0.0)), 0.25),
            (float(f.get("achievement_mentions_count", 0.0)), 0.20),
            (float(f.get("project_mentions_count", 0.0)), 0.20),
        ]
    )

    trust_items = [
        (float(f.get("completeness_score", 0.0)), 0.36),
        (float(f.get("consistency_score", 0.0)), 0.30),
        (float(f.get("evidence_count", 0.0)), 0.22),
        (float(f.get("behavioral_completion_score", 0.0)), 0.12),
    ]
    if use_semantic_layer:
        trust_items.append((float(f.get("semantic_authenticity_groundedness", 0.0)), 0.18))
    trust_base = weighted_average_normalized(trust_items)
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

    confidence_items = [
        (float(f.get("specificity_score", 0.0)), CONFIG.weights.confidence_components["specificity_score"]),
        (float(f.get("evidence_count", 0.0)), CONFIG.weights.confidence_components["evidence_count"]),
        (float(f.get("consistency_score", 0.0)), CONFIG.weights.confidence_components["consistency_score"]),
        (float(f.get("completeness_score", 0.0)), CONFIG.weights.confidence_components["completeness_score"]),
    ]
    if use_semantic_layer:
        confidence_items.append((float(f.get("semantic_authenticity_groundedness", 0.0)), 0.22))
    confidence_base = weighted_average_normalized(confidence_items)

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
