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


def _material_support_score(feature_map: dict[str, float | bool]) -> float:
    return weighted_average_normalized(
        [
            (float(feature_map.get("docs_count_score", 0.0)), 0.50),
            (float(feature_map.get("portfolio_links_score", 0.0)), 0.15),
            (1.0 if bool(feature_map.get("has_video_presentation", False)) else 0.0, 0.35),
        ]
    )


def _unsupported_narrative_penalty(feature_map: dict[str, float | bool], material_support: float) -> float:
    supported_signal = weighted_average_normalized(
        [
            (float(feature_map.get("evidence_richness", 0.0)), 0.45),
            (float(feature_map.get("specificity_score", 0.0)), 0.35),
            (material_support, 0.20),
        ]
    )
    gap = max(0.0, float(feature_map.get("program_fit", 0.0)) - supported_signal)
    return gap * 0.18


def _hidden_material_bonus(feature_map: dict[str, float | bool], material_support: float) -> float:
    hidden_signal = float(feature_map.get("semantic_hidden_potential", 0.0))
    specificity_gap = clamp01(1.0 - float(feature_map.get("specificity_score", 0.0)))
    evidence_signal = float(feature_map.get("evidence_count", 0.0))
    multimodal_bonus = hidden_signal * material_support * specificity_gap * 0.16
    text_only_bonus = hidden_signal * evidence_signal * specificity_gap * 0.05
    return min(0.08, multimodal_bonus + text_only_bonus)


def _committee_calibration_signal(feature_map: dict[str, float | bool], authenticity_risk_raw: float) -> tuple[float, dict[str, float]]:
    material_support = _material_support_score(feature_map)
    unsupported_narrative_penalty = _unsupported_narrative_penalty(feature_map, material_support)
    low_evidence_with_no_support_penalty = (
        0.08 if bool(feature_map.get("low_evidence_flag", False)) and material_support < 0.25 else 0.0
    )
    contradiction_penalty = 0.18 if bool(feature_map.get("contradiction_flag", False)) else 0.0
    genericness_penalty = float(feature_map.get("genericness_score", 0.0)) * 0.05
    polished_penalty = float(feature_map.get("polished_but_empty_score", 0.0)) * 0.10
    mismatch_penalty = float(feature_map.get("cross_section_mismatch_score", 0.0)) * 0.06
    authenticity_penalty = max(0.0, authenticity_risk_raw - 0.65) * 0.10

    base = weighted_average_normalized(
        [
            (float(feature_map.get("specificity_score", 0.0)), 0.14),
            (float(feature_map.get("consistency_score", 0.0)), 0.14),
            (float(feature_map.get("completeness_score", 0.0)), 0.14),
            (float(feature_map.get("evidence_richness", 0.0)), 0.12),
            (float(feature_map.get("docs_count_score", 0.0)), 0.12),
            (material_support, 0.14),
            (float(feature_map.get("growth_trajectory", 0.0)), 0.08),
            (float(feature_map.get("resilience", 0.0)), 0.07),
            (float(feature_map.get("evidence_count", 0.0)), 0.05),
            (float(feature_map.get("initiative", 0.0)), 0.03),
            (float(feature_map.get("semantic_authenticity_groundedness", 0.0)), 0.03),
            (float(feature_map.get("semantic_growth_trajectory", 0.0)), 0.04),
            (float(feature_map.get("program_fit", 0.0)), 0.02),
        ]
    )

    hidden_bonus = _hidden_material_bonus(feature_map, material_support)
    total_penalty = (
        contradiction_penalty
        + low_evidence_with_no_support_penalty
        + genericness_penalty
        + polished_penalty
        + mismatch_penalty
        + unsupported_narrative_penalty
        + authenticity_penalty
    )

    signal = clamp01(base + hidden_bonus - total_penalty)
    diagnostics = {
        "material_support_score": round(material_support, 6),
        "hidden_material_bonus": round(hidden_bonus, 6),
        "unsupported_narrative_penalty": round(unsupported_narrative_penalty, 6),
        "low_evidence_with_no_support_penalty": round(low_evidence_with_no_support_penalty, 6),
        "contradiction_penalty": round(contradiction_penalty, 6),
        "genericness_penalty": round(genericness_penalty, 6),
        "polished_penalty": round(polished_penalty, 6),
        "mismatch_penalty": round(mismatch_penalty, 6),
        "authenticity_penalty": round(authenticity_penalty, 6),
        "committee_signal_base": round(base, 6),
        "committee_signal": round(signal, 6),
    }
    return signal, diagnostics


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
    material_support = _material_support_score(f)

    potential_items = [
        ("growth_trajectory", float(f.get("growth_trajectory", 0.0)), 0.20),
        ("resilience", float(f.get("resilience", 0.0)), 0.16),
        ("initiative", float(f.get("initiative", 0.0)), 0.14),
        ("leadership_impact", float(f.get("leadership_impact", 0.0)), 0.10),
        ("evidence_richness", float(f.get("evidence_richness", 0.0)), 0.08),
        ("program_fit", float(f.get("program_fit", 0.0)), 0.06),
    ]
    if use_semantic_layer:
        potential_items.extend(
            [
                ("semantic_growth_trajectory", float(f.get("semantic_growth_trajectory", 0.0)), 0.14),
                ("semantic_leadership_potential", float(f.get("semantic_leadership_potential", 0.0)), 0.10),
                ("semantic_hidden_potential", float(f.get("semantic_hidden_potential", 0.0)), 0.12),
            ]
        )
    potential_rows, potential_raw = _component_rows(potential_items)

    motivation_items = [
        ("motivation_clarity", float(f.get("motivation_clarity", 0.0)), 0.18),
        ("program_fit", float(f.get("program_fit", 0.0)), 0.10),
        ("evidence_richness", float(f.get("evidence_richness", 0.0)), 0.16),
        ("specificity_score", float(f.get("specificity_score", 0.0)), 0.12),
        ("consistency_score", float(f.get("consistency_score", 0.0)), 0.10),
    ]
    if use_semantic_layer:
        motivation_items.extend(
            [
                ("semantic_motivation_authenticity", float(f.get("semantic_motivation_authenticity", 0.0)), 0.18),
                ("semantic_authenticity_groundedness", float(f.get("semantic_authenticity_groundedness", 0.0)), 0.16),
            ]
        )
    motivation_rows, motivation_raw = _component_rows(motivation_items)

    leadership_items = [
        ("initiative", float(f.get("initiative", 0.0)), 0.18),
        ("leadership_impact", float(f.get("leadership_impact", 0.0)), 0.16),
        ("evidence_richness", float(f.get("evidence_richness", 0.0)), 0.14),
        ("evidence_count", float(f.get("evidence_count", 0.0)), 0.10),
        ("material_support_score", material_support, 0.10),
    ]
    if use_semantic_layer:
        leadership_items.extend(
            [
                ("semantic_leadership_potential", float(f.get("semantic_leadership_potential", 0.0)), 0.18),
                ("semantic_hidden_potential", float(f.get("semantic_hidden_potential", 0.0)), 0.14),
            ]
        )
    leadership_rows, leadership_raw = _component_rows(leadership_items)

    experience_items = [
        ("specificity_score", float(f.get("specificity_score", 0.0)), 0.24),
        ("evidence_count", float(f.get("evidence_count", 0.0)), 0.18),
        ("evidence_richness", float(f.get("evidence_richness", 0.0)), 0.16),
        ("project_mentions_count", float(f.get("project_mentions_count", 0.0)), 0.08),
        ("docs_count_score", float(f.get("docs_count_score", 0.0)), 0.16),
        ("portfolio_links_score", float(f.get("portfolio_links_score", 0.0)), 0.06),
        ("has_video_presentation", 1.0 if bool(f.get("has_video_presentation", False)) else 0.0, 0.08),
        ("completeness_score", float(f.get("completeness_score", 0.0)), 0.04),
    ]
    experience_rows, experience_raw = _component_rows(experience_items)

    trust_items = [
        ("completeness_score", float(f.get("completeness_score", 0.0)), 0.22),
        ("consistency_score", float(f.get("consistency_score", 0.0)), 0.18),
        ("evidence_count", float(f.get("evidence_count", 0.0)), 0.12),
        ("behavioral_completion_score", float(f.get("behavioral_completion_score", 0.0)), 0.08),
        ("material_support_score", material_support, 0.16),
        ("docs_count_score", float(f.get("docs_count_score", 0.0)), 0.08),
    ]
    if use_semantic_layer:
        trust_items.append(("semantic_authenticity_groundedness", float(f.get("semantic_authenticity_groundedness", 0.0)), 0.16))
    trust_rows, trust_base = _component_rows(trust_items)

    unsupported_narrative_penalty = _unsupported_narrative_penalty(f, material_support)
    trust_penalty = 0.0
    trust_penalty_contrib = {
        "contradiction_flag": 0.18 if bool(f.get("contradiction_flag", False)) else 0.0,
        "low_evidence_flag": (0.14 if material_support < 0.25 else 0.05) if bool(f.get("low_evidence_flag", False)) else 0.0,
        "genericness_penalty": float(f.get("genericness_score", 0.0)) * 0.06,
        "polished_empty_penalty": float(f.get("polished_but_empty_score", 0.0)) * 0.08,
        "mismatch_penalty": float(f.get("cross_section_mismatch_score", 0.0)) * 0.06,
        "unsupported_narrative_penalty": unsupported_narrative_penalty,
    }
    trust_penalty += trust_penalty_contrib["contradiction_flag"]
    trust_penalty += trust_penalty_contrib["low_evidence_flag"]
    trust_penalty += trust_penalty_contrib["genericness_penalty"]
    trust_penalty += trust_penalty_contrib["polished_empty_penalty"]
    trust_penalty += trust_penalty_contrib["mismatch_penalty"]
    trust_penalty += trust_penalty_contrib["unsupported_narrative_penalty"]
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
    merit_rows, merit_base_raw = _component_rows(merit_components)

    committee_signal, committee_diagnostics = _committee_calibration_signal(f, authenticity_risk_raw)
    merit_raw = clamp01((merit_base_raw * 0.35) + (committee_signal * 0.65))

    confidence_items = [
        ("specificity_score", float(f.get("specificity_score", 0.0)), CONFIG.weights.confidence_components["specificity_score"]),
        ("evidence_count", float(f.get("evidence_count", 0.0)), CONFIG.weights.confidence_components["evidence_count"]),
        ("consistency_score", float(f.get("consistency_score", 0.0)), CONFIG.weights.confidence_components["consistency_score"]),
        ("completeness_score", float(f.get("completeness_score", 0.0)), CONFIG.weights.confidence_components["completeness_score"]),
        ("material_support_score", material_support, 0.18),
    ]
    if use_semantic_layer:
        confidence_items.append(("semantic_authenticity_groundedness", float(f.get("semantic_authenticity_groundedness", 0.0)), 0.10))
    confidence_rows, confidence_base = _component_rows(confidence_items)

    confidence_penalty = 0.0
    confidence_penalty_contrib = {
        "contradiction_flag": 0.15 if bool(f.get("contradiction_flag", False)) else 0.0,
        "low_evidence_flag": (0.10 if material_support < 0.25 else 0.04) if bool(f.get("low_evidence_flag", False)) else 0.0,
        "authenticity_risk_penalty": max(0.0, authenticity_risk_raw - 0.60) * 0.12,
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
            "committee_calibration_signal": "clamp01(weighted_average(priority_features) + hidden_bonus - calibration_penalties)",
            "merit": "clamp01(0.35 * merit_base + 0.65 * committee_calibration_signal)",
            "confidence": "clamp01(weighted_average(confidence_features) - confidence_penalty)",
        },
        "components": {
            "potential": potential_rows,
            "motivation": motivation_rows,
            "leadership_agency": leadership_rows,
            "experience_skills": experience_rows,
            "trust_completeness_base": trust_rows,
            "merit_base": merit_rows,
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
            "committee_calibration": committee_diagnostics,
        },
        "axis_raw": {k: round(v, 6) for k, v in merit_breakdown.items()},
        "outputs": {
            "merit_raw": round(merit_raw, 6),
            "merit_base_raw": round(merit_base_raw, 6),
            "committee_signal_raw": round(committee_signal, 6),
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
    material_support = _material_support_score(f)

    potential_items = [
        (float(f.get("growth_trajectory", 0.0)), 0.20),
        (float(f.get("resilience", 0.0)), 0.16),
        (float(f.get("initiative", 0.0)), 0.14),
        (float(f.get("leadership_impact", 0.0)), 0.10),
        (float(f.get("evidence_richness", 0.0)), 0.08),
        (float(f.get("program_fit", 0.0)), 0.06),
    ]
    if use_semantic_layer:
        potential_items.extend(
            [
                (float(f.get("semantic_growth_trajectory", 0.0)), 0.14),
                (float(f.get("semantic_leadership_potential", 0.0)), 0.10),
                (float(f.get("semantic_hidden_potential", 0.0)), 0.12),
            ]
        )
    potential = weighted_average_normalized(potential_items)

    motivation_items = [
        (float(f.get("motivation_clarity", 0.0)), 0.18),
        (float(f.get("program_fit", 0.0)), 0.10),
        (float(f.get("evidence_richness", 0.0)), 0.16),
        (float(f.get("specificity_score", 0.0)), 0.12),
        (float(f.get("consistency_score", 0.0)), 0.10),
    ]
    if use_semantic_layer:
        motivation_items.extend(
            [
                (float(f.get("semantic_motivation_authenticity", 0.0)), 0.18),
                (float(f.get("semantic_authenticity_groundedness", 0.0)), 0.16),
            ]
        )
    motivation = weighted_average_normalized(motivation_items)

    leadership_items = [
        (float(f.get("initiative", 0.0)), 0.18),
        (float(f.get("leadership_impact", 0.0)), 0.16),
        (float(f.get("evidence_richness", 0.0)), 0.14),
        (float(f.get("evidence_count", 0.0)), 0.10),
        (material_support, 0.10),
    ]
    if use_semantic_layer:
        leadership_items.extend(
            [
                (float(f.get("semantic_leadership_potential", 0.0)), 0.18),
                (float(f.get("semantic_hidden_potential", 0.0)), 0.14),
            ]
        )
    leadership_agency = weighted_average_normalized(leadership_items)
    experience_skills = weighted_average_normalized(
        [
            (float(f.get("specificity_score", 0.0)), 0.24),
            (float(f.get("evidence_count", 0.0)), 0.18),
            (float(f.get("evidence_richness", 0.0)), 0.16),
            (float(f.get("project_mentions_count", 0.0)), 0.08),
            (float(f.get("docs_count_score", 0.0)), 0.16),
            (float(f.get("portfolio_links_score", 0.0)), 0.06),
            (1.0 if bool(f.get("has_video_presentation", False)) else 0.0, 0.08),
            (float(f.get("completeness_score", 0.0)), 0.04),
        ]
    )

    trust_items = [
        (float(f.get("completeness_score", 0.0)), 0.22),
        (float(f.get("consistency_score", 0.0)), 0.18),
        (float(f.get("evidence_count", 0.0)), 0.12),
        (float(f.get("behavioral_completion_score", 0.0)), 0.08),
        (material_support, 0.16),
        (float(f.get("docs_count_score", 0.0)), 0.08),
    ]
    if use_semantic_layer:
        trust_items.append((float(f.get("semantic_authenticity_groundedness", 0.0)), 0.16))
    trust_base = weighted_average_normalized(trust_items)
    trust_penalty = 0.0
    if bool(f.get("contradiction_flag", False)):
        trust_penalty += 0.18
    if bool(f.get("low_evidence_flag", False)):
        trust_penalty += 0.14 if material_support < 0.25 else 0.05
    trust_penalty += float(f.get("genericness_score", 0.0)) * 0.06
    trust_penalty += float(f.get("polished_but_empty_score", 0.0)) * 0.08
    trust_penalty += float(f.get("cross_section_mismatch_score", 0.0)) * 0.06
    trust_penalty += _unsupported_narrative_penalty(f, material_support)
    trust_completeness = clamp01(trust_base - trust_penalty)

    merit_breakdown_raw = {
        "potential": potential,
        "motivation": motivation,
        "leadership_agency": leadership_agency,
        "experience_skills": experience_skills,
        "trust_completeness": trust_completeness,
    }

    merit_base_raw = weighted_average_normalized(
        [(value, CONFIG.weights.merit_breakdown[key]) for key, value in merit_breakdown_raw.items()]
    )
    committee_signal, _committee_diagnostics = _committee_calibration_signal(f, authenticity_risk_raw)
    merit_raw = clamp01((merit_base_raw * 0.35) + (committee_signal * 0.65))

    confidence_items = [
        (float(f.get("specificity_score", 0.0)), CONFIG.weights.confidence_components["specificity_score"]),
        (float(f.get("evidence_count", 0.0)), CONFIG.weights.confidence_components["evidence_count"]),
        (float(f.get("consistency_score", 0.0)), CONFIG.weights.confidence_components["consistency_score"]),
        (float(f.get("completeness_score", 0.0)), CONFIG.weights.confidence_components["completeness_score"]),
        (material_support, 0.18),
    ]
    if use_semantic_layer:
        confidence_items.append((float(f.get("semantic_authenticity_groundedness", 0.0)), 0.10))
    confidence_base = weighted_average_normalized(confidence_items)

    confidence_penalty = 0.0
    if bool(f.get("contradiction_flag", False)):
        confidence_penalty += 0.15
    if bool(f.get("low_evidence_flag", False)):
        confidence_penalty += 0.10 if material_support < 0.25 else 0.04
    confidence_penalty += max(0.0, authenticity_risk_raw - 0.60) * 0.12

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
