"""Scoring engine for merit, confidence, and committee-facing score traces."""

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


def _component_raw(items: list[tuple[str, float, float]]) -> float:
    return weighted_average_normalized([(value, weight) for _, value, weight in items])


def _source_support_score(feature_map: dict[str, float | bool]) -> float:
    """Confidence-only support signal based on answer completeness, not resources."""
    return weighted_average_normalized(
        [
            (float(feature_map.get("text_completeness_score", 0.0)), 0.42),
            (float(feature_map.get("question_coverage_score", 0.0)), 0.24),
            (float(feature_map.get("behavioral_completion_score", 0.0)), 0.22),
            (float(feature_map.get("consistency_score", 0.0)), 0.12),
        ]
    )


def _practical_action_score(feature_map: dict[str, float | bool]) -> float:
    return weighted_average_normalized(
        [
            (float(feature_map.get("initiative", 0.0)), 0.24),
            (float(feature_map.get("leadership_impact", 0.0)), 0.18),
            (float(feature_map.get("evidence_count", 0.0)), 0.18),
            (float(feature_map.get("project_mentions_count", 0.0)), 0.14),
            (float(feature_map.get("trajectory_adaptation_score", 0.0)), 0.14),
            (float(feature_map.get("trajectory_outcome_score", 0.0)), 0.12),
        ]
    )


def _language_fairness_discount(feature_map: dict[str, float | bool]) -> float:
    coherence = weighted_average_normalized(
        [
            (float(feature_map.get("evidence_count", 0.0)), 0.34),
            (float(feature_map.get("specificity_score", 0.0)), 0.16),
            (float(feature_map.get("consistency_score", 0.0)), 0.28),
            (1.0 - float(feature_map.get("cross_section_mismatch_score", 0.0)), 0.22),
        ]
    )
    if coherence < 0.32:
        return 0.0
    support_quality = weighted_average_normalized(
        [
            (float(feature_map.get("section_claim_overlap_score", 0.0)), 0.40),
            (float(feature_map.get("section_role_consistency_score", 0.0)), 0.30),
            (float(feature_map.get("section_time_consistency_score", 0.0)), 0.30),
        ]
    )
    return clamp01((coherence * 0.65) + (support_quality * 0.35)) * 0.18


def _unsupported_narrative_penalty(feature_map: dict[str, float | bool]) -> float:
    practical_action = _practical_action_score(feature_map)
    supported_signal = weighted_average_normalized(
        [
            (float(feature_map.get("evidence_richness", 0.0)), 0.34),
            (float(feature_map.get("specificity_score", 0.0)), 0.28),
            (float(feature_map.get("evidence_count", 0.0)), 0.24),
            (practical_action, 0.14),
        ]
    )
    narrative_signal = weighted_average_normalized(
        [
            (float(feature_map.get("program_fit", 0.0)), 0.48),
            (float(feature_map.get("motivation_clarity", 0.0)), 0.32),
            (float(feature_map.get("community_value_orientation", 0.0)), 0.20),
        ]
    )
    gap = max(0.0, narrative_signal - supported_signal)
    return gap * 0.18


def _hidden_signal_bonus(feature_map: dict[str, float | bool]) -> float:
    hidden_signal = float(feature_map.get("semantic_hidden_potential", 0.0))
    specificity_gap = clamp01(1.0 - float(feature_map.get("specificity_score", 0.0)))
    evidence_signal = float(feature_map.get("evidence_count", 0.0))
    practical_action = _practical_action_score(feature_map)
    trajectory_signal = weighted_average_normalized(
        [
            (float(feature_map.get("growth_trajectory", 0.0)), 0.36),
            (float(feature_map.get("trajectory_adaptation_score", 0.0)), 0.34),
            (float(feature_map.get("trajectory_reflection_score", 0.0)), 0.30),
        ]
    )
    text_only_bonus = hidden_signal * evidence_signal * specificity_gap * 0.09
    action_bonus = hidden_signal * practical_action * specificity_gap * 0.07
    trajectory_bonus = hidden_signal * trajectory_signal * specificity_gap * 0.06
    return min(0.07, text_only_bonus + action_bonus + trajectory_bonus)


def _thin_evidence_gap(feature_map: dict[str, float | bool]) -> float:
    evidence_floor = weighted_average_normalized(
        [
            (float(feature_map.get("evidence_count", 0.0)), 0.55),
            (float(feature_map.get("evidence_richness", 0.0)), 0.25),
            (float(feature_map.get("specificity_score", 0.0)), 0.20),
        ]
    )
    return max(0.0, 0.38 - evidence_floor)


def _committee_calibration_signal(
    feature_map: dict[str, float | bool],
    authenticity_risk_raw: float,
) -> tuple[float, dict[str, float]]:
    source_support = _source_support_score(feature_map)
    practical_action = _practical_action_score(feature_map)
    fairness_discount = _language_fairness_discount(feature_map)
    thin_evidence_gap = _thin_evidence_gap(feature_map)
    unsupported_narrative_penalty = _unsupported_narrative_penalty(feature_map)
    low_evidence_with_low_support_penalty = 0.0
    if bool(feature_map.get("low_evidence_flag", False)) and source_support < 0.35:
        growth_discount = float(feature_map.get("growth_trajectory", 0.0)) * 0.08
        action_discount = practical_action * 0.12
        low_evidence_with_low_support_penalty = max(0.025, 0.08 - growth_discount - action_discount)
    contradiction_penalty = 0.18 if bool(feature_map.get("contradiction_flag", False)) else 0.0
    genericness_penalty = float(feature_map.get("genericness_score", 0.0)) * 0.05 * (1.0 - fairness_discount)
    polished_penalty = float(feature_map.get("polished_but_empty_score", 0.0)) * (
        0.08 + (thin_evidence_gap * 0.18)
    ) * (1.0 - (fairness_discount * 0.85))
    mismatch_penalty = float(feature_map.get("cross_section_mismatch_score", 0.0)) * 0.06
    authenticity_penalty = max(0.0, authenticity_risk_raw - 0.65) * 0.10

    base = weighted_average_normalized(
        [
            (source_support, 0.12),
            (float(feature_map.get("evidence_richness", 0.0)), 0.16),
            (float(feature_map.get("evidence_count", 0.0)), 0.10),
            (practical_action, 0.18),
            (float(feature_map.get("growth_trajectory", 0.0)), 0.14),
            (float(feature_map.get("trajectory_reflection_score", 0.0)), 0.08),
            (float(feature_map.get("resilience", 0.0)), 0.08),
            (float(feature_map.get("community_value_orientation", 0.0)), 0.08),
            (float(feature_map.get("semantic_authenticity_groundedness", 0.0)), 0.03),
            (float(feature_map.get("semantic_growth_trajectory", 0.0)), 0.03),
        ]
    )

    hidden_bonus = _hidden_signal_bonus(feature_map)
    total_penalty = (
        contradiction_penalty
        + low_evidence_with_low_support_penalty
        + genericness_penalty
        + polished_penalty
        + mismatch_penalty
        + unsupported_narrative_penalty
        + authenticity_penalty
    )

    signal = clamp01(base + hidden_bonus - total_penalty)
    diagnostics = {
        "source_support_score": round(source_support, 6),
        "hidden_signal_bonus": round(hidden_bonus, 6),
        "practical_action_score": round(practical_action, 6),
        "thin_evidence_gap": round(thin_evidence_gap, 6),
        "unsupported_narrative_penalty": round(unsupported_narrative_penalty, 6),
        "low_evidence_with_low_support_penalty": round(low_evidence_with_low_support_penalty, 6),
        "contradiction_penalty": round(contradiction_penalty, 6),
        "genericness_penalty": round(genericness_penalty, 6),
        "polished_penalty": round(polished_penalty, 6),
        "mismatch_penalty": round(mismatch_penalty, 6),
        "authenticity_penalty": round(authenticity_penalty, 6),
        "committee_signal_base": round(base, 6),
        "committee_signal": round(signal, 6),
        "language_fairness_discount": round(fairness_discount, 6),
    }
    return signal, diagnostics


def _potential_items(feature_map: dict[str, float | bool], use_semantic_layer: bool) -> list[tuple[str, float, float]]:
    items = [
        ("growth_trajectory", float(feature_map.get("growth_trajectory", 0.0)), 0.20),
        ("resilience", float(feature_map.get("resilience", 0.0)), 0.16),
        ("trajectory_adaptation_score", float(feature_map.get("trajectory_adaptation_score", 0.0)), 0.12),
        ("trajectory_reflection_score", float(feature_map.get("trajectory_reflection_score", 0.0)), 0.10),
        ("initiative", float(feature_map.get("initiative", 0.0)), 0.10),
        ("leadership_impact", float(feature_map.get("leadership_impact", 0.0)), 0.08),
        ("evidence_richness", float(feature_map.get("evidence_richness", 0.0)), 0.08),
    ]
    if use_semantic_layer:
        items.extend(
            [
                ("semantic_growth_trajectory", float(feature_map.get("semantic_growth_trajectory", 0.0)), 0.08),
                ("semantic_leadership_potential", float(feature_map.get("semantic_leadership_potential", 0.0)), 0.06),
                ("semantic_hidden_potential", float(feature_map.get("semantic_hidden_potential", 0.0)), 0.10),
            ]
        )
    return items


def _motivation_items(feature_map: dict[str, float | bool], use_semantic_layer: bool) -> list[tuple[str, float, float]]:
    items = [
        ("motivation_clarity", float(feature_map.get("motivation_clarity", 0.0)), 0.20),
        ("program_fit", float(feature_map.get("program_fit", 0.0)), 0.12),
        ("evidence_richness", float(feature_map.get("evidence_richness", 0.0)), 0.16),
        ("specificity_score", float(feature_map.get("specificity_score", 0.0)), 0.12),
        ("consistency_score", float(feature_map.get("consistency_score", 0.0)), 0.10),
    ]
    if use_semantic_layer:
        items.extend(
            [
                ("semantic_motivation_authenticity", float(feature_map.get("semantic_motivation_authenticity", 0.0)), 0.18),
                ("semantic_authenticity_groundedness", float(feature_map.get("semantic_authenticity_groundedness", 0.0)), 0.12),
            ]
        )
    return items


def _leadership_items(
    feature_map: dict[str, float | bool],
    use_semantic_layer: bool,
) -> list[tuple[str, float, float]]:
    items = [
        ("initiative", float(feature_map.get("initiative", 0.0)), 0.20),
        ("leadership_impact", float(feature_map.get("leadership_impact", 0.0)), 0.18),
        ("evidence_richness", float(feature_map.get("evidence_richness", 0.0)), 0.14),
        ("evidence_count", float(feature_map.get("evidence_count", 0.0)), 0.10),
        ("project_mentions_count", float(feature_map.get("project_mentions_count", 0.0)), 0.10),
        ("trajectory_outcome_score", float(feature_map.get("trajectory_outcome_score", 0.0)), 0.08),
        ("community_value_orientation", float(feature_map.get("community_value_orientation", 0.0)), 0.06),
    ]
    if use_semantic_layer:
        items.extend(
            [
                ("semantic_leadership_potential", float(feature_map.get("semantic_leadership_potential", 0.0)), 0.14),
                ("semantic_hidden_potential", float(feature_map.get("semantic_hidden_potential", 0.0)), 0.10),
            ]
        )
    return items


def _community_items(feature_map: dict[str, float | bool], use_semantic_layer: bool) -> list[tuple[str, float, float]]:
    items = [
        ("community_value_orientation", float(feature_map.get("community_value_orientation", 0.0)), 0.34),
        ("program_fit", float(feature_map.get("program_fit", 0.0)), 0.14),
        ("motivation_clarity", float(feature_map.get("motivation_clarity", 0.0)), 0.10),
        ("leadership_impact", float(feature_map.get("leadership_impact", 0.0)), 0.10),
        ("initiative", float(feature_map.get("initiative", 0.0)), 0.10),
        ("evidence_richness", float(feature_map.get("evidence_richness", 0.0)), 0.12),
    ]
    if use_semantic_layer:
        items.extend(
            [
                ("semantic_community_orientation", float(feature_map.get("semantic_community_orientation", 0.0)), 0.10),
            ]
        )
    return items


def _experience_items(feature_map: dict[str, float | bool]) -> list[tuple[str, float, float]]:
    return [
        ("certificate_score_normalized", float(feature_map.get("certificate_score_normalized", 0.5)), 0.20),
        ("english_score_normalized", float(feature_map.get("english_score_normalized", 0.5)), 0.18),
        ("specificity_score", float(feature_map.get("specificity_score", 0.0)), 0.24),
        ("evidence_count", float(feature_map.get("evidence_count", 0.0)), 0.14),
        ("evidence_richness", float(feature_map.get("evidence_richness", 0.0)), 0.10),
        ("project_mentions_count", float(feature_map.get("project_mentions_count", 0.0)), 0.10),
        ("achievement_mentions_count", float(feature_map.get("achievement_mentions_count", 0.0)), 0.08),
        ("trajectory_outcome_score", float(feature_map.get("trajectory_outcome_score", 0.0)), 0.10),
        ("trajectory_adaptation_score", float(feature_map.get("trajectory_adaptation_score", 0.0)), 0.10),
    ]


def _trust_items(
    feature_map: dict[str, float | bool],
    use_semantic_layer: bool,
    source_support: float,
) -> list[tuple[str, float, float]]:
    items = [
        ("completeness_score", float(feature_map.get("completeness_score", 0.0)), 0.24),
        ("consistency_score", float(feature_map.get("consistency_score", 0.0)), 0.24),
        ("specificity_score", float(feature_map.get("specificity_score", 0.0)), 0.10),
        ("evidence_count", float(feature_map.get("evidence_count", 0.0)), 0.16),
        ("behavioral_completion_score", float(feature_map.get("behavioral_completion_score", 0.0)), 0.10),
        ("source_support_score", source_support, 0.08),
        ("text_completeness_score", float(feature_map.get("text_completeness_score", 0.0)), 0.08),
    ]
    if use_semantic_layer:
        items.append(("semantic_authenticity_groundedness", float(feature_map.get("semantic_authenticity_groundedness", 0.0)), 0.08))
    return items


def _confidence_items(
    feature_map: dict[str, float | bool],
    use_semantic_layer: bool,
    source_support: float,
) -> list[tuple[str, float, float]]:
    items = [
        ("specificity_score", float(feature_map.get("specificity_score", 0.0)), CONFIG.weights.confidence_components["specificity_score"]),
        ("evidence_count", float(feature_map.get("evidence_count", 0.0)), CONFIG.weights.confidence_components["evidence_count"]),
        ("consistency_score", float(feature_map.get("consistency_score", 0.0)), CONFIG.weights.confidence_components["consistency_score"]),
        ("completeness_score", float(feature_map.get("completeness_score", 0.0)), CONFIG.weights.confidence_components["completeness_score"]),
        ("source_support_score", source_support, 0.08),
    ]
    if use_semantic_layer:
        items.append(("semantic_authenticity_groundedness", float(feature_map.get("semantic_authenticity_groundedness", 0.0)), 0.10))
    return items


def _trust_penalty_components(
    feature_map: dict[str, float | bool],
    *,
    source_support: float,
    fairness_discount: float,
) -> dict[str, float]:
    thin_evidence_gap = _thin_evidence_gap(feature_map)
    practical_action = _practical_action_score(feature_map)
    low_evidence_penalty = 0.0
    if bool(feature_map.get("low_evidence_flag", False)):
        base_penalty = 0.14 if source_support < 0.35 else 0.05
        growth_discount = float(feature_map.get("growth_trajectory", 0.0)) * 0.05
        action_discount = practical_action * 0.08
        floor = 0.03 if source_support < 0.35 else 0.02
        low_evidence_penalty = max(floor, base_penalty - growth_discount - action_discount)
    return {
        "contradiction_flag": 0.18 if bool(feature_map.get("contradiction_flag", False)) else 0.0,
        "low_evidence_flag": low_evidence_penalty,
        "genericness_penalty": float(feature_map.get("genericness_score", 0.0)) * 0.06 * (1.0 - fairness_discount),
        "polished_empty_penalty": float(feature_map.get("polished_but_empty_score", 0.0))
        * (0.06 + (thin_evidence_gap * 0.16))
        * (1.0 - (fairness_discount * 0.85)),
        "mismatch_penalty": float(feature_map.get("cross_section_mismatch_score", 0.0)) * 0.06,
        "unsupported_narrative_penalty": _unsupported_narrative_penalty(feature_map),
    }


def _confidence_penalty_components(
    feature_map: dict[str, float | bool],
    *,
    source_support: float,
    authenticity_risk_raw: float,
) -> dict[str, float]:
    return {
        "contradiction_flag": 0.15 if bool(feature_map.get("contradiction_flag", False)) else 0.0,
        "low_evidence_flag": (0.10 if source_support < 0.35 else 0.04) if bool(feature_map.get("low_evidence_flag", False)) else 0.0,
        "authenticity_risk_penalty": max(0.0, authenticity_risk_raw - 0.60) * 0.12,
    }


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
    source_support = _source_support_score(f)
    fairness_discount = _language_fairness_discount(f)

    potential_items = _potential_items(f, use_semantic_layer)
    potential_rows, potential_raw = _component_rows(potential_items)

    motivation_items = _motivation_items(f, use_semantic_layer)
    motivation_rows, motivation_raw = _component_rows(motivation_items)

    leadership_items = _leadership_items(f, use_semantic_layer)
    leadership_rows, leadership_raw = _component_rows(leadership_items)

    community_items = _community_items(f, use_semantic_layer)
    community_rows, community_raw = _component_rows(community_items)

    experience_items = _experience_items(f)
    experience_rows, experience_raw = _component_rows(experience_items)

    trust_items = _trust_items(f, use_semantic_layer, source_support)
    trust_rows, trust_base = _component_rows(trust_items)
    trust_penalty_contrib = _trust_penalty_components(f, source_support=source_support, fairness_discount=fairness_discount)
    trust_penalty = sum(trust_penalty_contrib.values())
    trust_raw = clamp01(trust_base - trust_penalty)

    merit_breakdown = {
        "potential": potential_raw,
        "motivation": motivation_raw,
        "leadership_agency": leadership_raw,
        "community_values": community_raw,
        "experience_skills": experience_raw,
        "trust_completeness": trust_raw,
    }

    merit_axes = list(CONFIG.weights.merit_breakdown.keys())
    merit_components = [(axis, merit_breakdown[axis], CONFIG.weights.merit_breakdown[axis]) for axis in merit_axes]
    merit_rows, merit_base_raw = _component_rows(merit_components)

    committee_signal, committee_diagnostics = _committee_calibration_signal(f, authenticity_risk_raw)
    merit_raw = clamp01((merit_base_raw * 0.72) + (committee_signal * 0.28))

    confidence_items = _confidence_items(f, use_semantic_layer, source_support)
    confidence_rows, confidence_base = _component_rows(confidence_items)
    confidence_penalty_contrib = _confidence_penalty_components(
        f,
        source_support=source_support,
        authenticity_risk_raw=authenticity_risk_raw,
    )
    confidence_penalty = sum(confidence_penalty_contrib.values())
    confidence_raw = clamp01(confidence_base - confidence_penalty)

    return {
        "formulas": {
            "potential": "weighted_average([growth_trajectory,resilience,trajectory_adaptation_score,trajectory_reflection_score,initiative,leadership_impact,evidence_richness,semantic_growth_trajectory?,semantic_leadership_potential?,semantic_hidden_potential?])",
            "motivation": "weighted_average([motivation_clarity,program_fit,evidence_richness,specificity_score,semantic_motivation_authenticity?])",
            "leadership_agency": "weighted_average([initiative,leadership_impact,evidence_richness,evidence_count,project_mentions_count,semantic_leadership_potential?,semantic_hidden_potential?])",
            "community_values": "weighted_average([community_value_orientation,program_fit,motivation_clarity,leadership_impact,evidence_richness,semantic_community_orientation?])",
            "experience_skills": "weighted_average([specificity_score,evidence_count,evidence_richness,project_mentions_count,trajectory_outcome_score,trajectory_adaptation_score])",
            "trust_completeness": "clamp01(weighted_average(trust_features) - trust_penalty)",
            "committee_calibration_signal": "clamp01(weighted_average(priority_features) + hidden_bonus - calibration_penalties)",
            "merit": "clamp01(0.72 * merit_base + 0.28 * committee_calibration_signal)",
            "confidence": "clamp01(weighted_average(confidence_features) - confidence_penalty)",
        },
        "components": {
            "potential": potential_rows,
            "motivation": motivation_rows,
            "leadership_agency": leadership_rows,
            "community_values": community_rows,
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
    source_support = _source_support_score(f)
    fairness_discount = _language_fairness_discount(f)

    potential = _component_raw(_potential_items(f, use_semantic_layer))
    motivation = _component_raw(_motivation_items(f, use_semantic_layer))
    leadership_agency = _component_raw(_leadership_items(f, use_semantic_layer))
    community_values = _component_raw(_community_items(f, use_semantic_layer))
    experience_skills = _component_raw(_experience_items(f))
    trust_base = _component_raw(_trust_items(f, use_semantic_layer, source_support))
    trust_penalty = sum(_trust_penalty_components(f, source_support=source_support, fairness_discount=fairness_discount).values())
    trust_completeness = clamp01(trust_base - trust_penalty)

    merit_breakdown_raw = {
        "potential": potential,
        "motivation": motivation,
        "leadership_agency": leadership_agency,
        "community_values": community_values,
        "experience_skills": experience_skills,
        "trust_completeness": trust_completeness,
    }

    merit_base_raw = weighted_average_normalized(
        [(value, CONFIG.weights.merit_breakdown[key]) for key, value in merit_breakdown_raw.items()]
    )
    committee_signal, _committee_diagnostics = _committee_calibration_signal(f, authenticity_risk_raw)
    merit_raw = clamp01((merit_base_raw * 0.72) + (committee_signal * 0.28))

    confidence_base = _component_raw(_confidence_items(f, use_semantic_layer, source_support))
    confidence_penalty = sum(
        _confidence_penalty_components(
            f,
            source_support=source_support,
            authenticity_risk_raw=authenticity_risk_raw,
        ).values()
    )

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
