"""Transparent shortlist-oriented derived signals for committee workflow."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.decision import Recommendation
from app.utils.math_utils import clamp01, to_display_score, weighted_average_normalized


@dataclass(slots=True)
class ShortlistSignals:
    hidden_potential_score: int
    support_needed_score: int
    shortlist_priority_score: int
    evidence_coverage_score: int
    trajectory_score: int


@dataclass(slots=True)
class BatchShortlistSummary:
    ranked_candidate_ids: list[str]
    shortlist_candidate_ids: list[str]
    hidden_potential_candidate_ids: list[str]
    support_needed_candidate_ids: list[str]
    authenticity_review_candidate_ids: list[str]


def _pairwise_preference_score(left: object, right: object) -> float:
    """Transparent head-to-head comparison for shortlist ordering."""
    score = 0.0
    score += (getattr(left, "shortlist_priority_score", 0) - getattr(right, "shortlist_priority_score", 0)) * 0.42
    score += (getattr(left, "hidden_potential_score", 0) - getattr(right, "hidden_potential_score", 0)) * 0.18
    score += (getattr(left, "trajectory_score", 0) - getattr(right, "trajectory_score", 0)) * 0.12
    score += (getattr(left, "evidence_coverage_score", 0) - getattr(right, "evidence_coverage_score", 0)) * 0.11
    score += (getattr(left, "merit_score", 0) - getattr(right, "merit_score", 0)) * 0.09
    score += (getattr(left, "confidence_score", 0) - getattr(right, "confidence_score", 0)) * 0.05
    score += (getattr(right, "authenticity_risk", 100) - getattr(left, "authenticity_risk", 100)) * 0.08
    return score


def _pairwise_rank_results(results: list[object]) -> list[object]:
    if len(results) <= 1:
        return list(results)

    pairwise_totals: dict[str, float] = {getattr(item, "candidate_id", ""): 0.0 for item in results}
    for idx, left in enumerate(results):
        for jdx, right in enumerate(results):
            if idx >= jdx:
                continue
            preference = _pairwise_preference_score(left, right)
            left_id = getattr(left, "candidate_id", "")
            right_id = getattr(right, "candidate_id", "")
            if preference > 0:
                pairwise_totals[left_id] += 1.0 + min(0.5, preference / 100.0)
                pairwise_totals[right_id] -= min(0.5, preference / 120.0)
            elif preference < 0:
                magnitude = abs(preference)
                pairwise_totals[right_id] += 1.0 + min(0.5, magnitude / 100.0)
                pairwise_totals[left_id] -= min(0.5, magnitude / 120.0)

    return sorted(
        results,
        key=lambda item: (
            pairwise_totals.get(getattr(item, "candidate_id", ""), 0.0),
            getattr(item, "shortlist_priority_score", 0),
            getattr(item, "hidden_potential_score", 0),
            getattr(item, "trajectory_score", 0),
            getattr(item, "merit_score", 0),
            -getattr(item, "authenticity_risk", 100),
            getattr(item, "confidence_score", 0),
        ),
        reverse=True,
    )


def _underlying_signal_raw(
    *,
    feature_map: dict[str, float | bool | int | None],
    semantic_scores: dict[str, float | int],
) -> float:
    trajectory_signal = weighted_average_normalized(
        [
            (float(feature_map.get("trajectory_challenge_score", 0.0)), 0.22),
            (float(feature_map.get("trajectory_adaptation_score", 0.0)), 0.26),
            (float(feature_map.get("trajectory_reflection_score", 0.0)), 0.24),
            (float(feature_map.get("trajectory_outcome_score", 0.0)), 0.10),
            (float(feature_map.get("growth_trajectory", 0.0)), 0.18),
        ]
    )
    return weighted_average_normalized(
        [
            (trajectory_signal, 0.22),
            (float(feature_map.get("resilience", 0.0)), 0.11),
            (float(feature_map.get("initiative", 0.0)), 0.08),
            (float(feature_map.get("leadership_impact", 0.0)), 0.06),
            (float(semantic_scores.get("hidden_potential", 0.0)) / 100.0, 0.22),
            (float(semantic_scores.get("growth_trajectory", 0.0)) / 100.0, 0.18),
            (float(semantic_scores.get("leadership_potential", 0.0)) / 100.0, 0.13),
        ]
    )


def _self_presentation_raw(
    *,
    feature_map: dict[str, float | bool | int | None],
    semantic_scores: dict[str, float | int],
) -> float:
    return weighted_average_normalized(
        [
            (float(feature_map.get("motivation_clarity", 0.0)), 0.28),
            (float(feature_map.get("specificity_score", 0.0)), 0.24),
            (float(feature_map.get("evidence_richness", 0.0)), 0.18),
            (float(feature_map.get("evidence_count", 0.0)), 0.14),
            (float(feature_map.get("completeness_score", 0.0)), 0.08),
            (float(semantic_scores.get("motivation_authenticity", 0.0)) / 100.0, 0.08),
        ]
    )


def _hidden_potential_raw(
    *,
    feature_map: dict[str, float | bool | int | None],
    semantic_scores: dict[str, float | int],
    authenticity_risk: int,
) -> float:
    underlying_signal = _underlying_signal_raw(feature_map=feature_map, semantic_scores=semantic_scores)
    trajectory_signal = _trajectory_raw(feature_map=feature_map, semantic_scores=semantic_scores)
    self_presentation = _self_presentation_raw(feature_map=feature_map, semantic_scores=semantic_scores)
    overstatement_risk = weighted_average_normalized(
        [
            (float(feature_map.get("genericness_score", 0.0)), 0.18),
            (float(feature_map.get("polished_but_empty_score", 0.0)), 0.30),
            (float(feature_map.get("cross_section_mismatch_score", 0.0)), 0.22),
            (1.0 - float(feature_map.get("specificity_score", 0.0)), 0.15),
            (1.0 - float(feature_map.get("evidence_count", 0.0)), 0.15),
        ]
    )
    evidence_floor = weighted_average_normalized(
        [
            (float(feature_map.get("evidence_count", 0.0)), 0.34),
            (float(feature_map.get("evidence_richness", 0.0)), 0.22),
            (float(feature_map.get("consistency_score", 0.0)), 0.24),
            (float(feature_map.get("completeness_score", 0.0)), 0.20),
        ]
    )
    credible_action_signal = weighted_average_normalized(
        [
            (float(feature_map.get("trajectory_adaptation_score", 0.0)), 0.28),
            (float(feature_map.get("trajectory_reflection_score", 0.0)), 0.24),
            (float(feature_map.get("trajectory_outcome_score", 0.0)), 0.12),
            (float(feature_map.get("evidence_count", 0.0)), 0.22),
            (float(feature_map.get("initiative", 0.0)), 0.14),
        ]
    )
    understatement_gap = clamp01(underlying_signal - self_presentation)
    modest_presentation = clamp01(0.62 - self_presentation)
    low_evidence_penalty = max(0.0, 0.22 - float(feature_map.get("evidence_count", 0.0)))

    raw = (
        (underlying_signal * 0.38)
        + (trajectory_signal * 0.16)
        + (credible_action_signal * 0.18)
        + (understatement_gap * 0.18)
        + (evidence_floor * 0.10)
        + (modest_presentation * 0.08)
    )
    raw -= overstatement_risk * 0.08
    raw -= low_evidence_penalty * 0.16
    raw -= max(0.0, (authenticity_risk / 100.0) - 0.78) * 0.08
    return clamp01(raw)


def _support_needed_raw(
    *,
    feature_map: dict[str, float | bool | int | None],
    merit_score: int,
    confidence_score: int,
) -> float:
    support_gap = clamp01(1.0 - (confidence_score / 100.0))
    adaptation_signal = weighted_average_normalized(
        [
            (float(feature_map.get("growth_trajectory", 0.0)), 0.22),
            (float(feature_map.get("resilience", 0.0)), 0.20),
            (float(feature_map.get("trajectory_adaptation_score", 0.0)), 0.20),
            (float(feature_map.get("trajectory_reflection_score", 0.0)), 0.16),
            (float(feature_map.get("completeness_score", 0.0)), 0.12),
            (float(feature_map.get("consistency_score", 0.0)), 0.10),
        ]
    )
    promise_floor = clamp01(merit_score / 100.0)
    raw = (support_gap * 0.50) + (adaptation_signal * 0.20) + (promise_floor * 0.30)
    return clamp01(raw)


def _evidence_coverage_raw(feature_map: dict[str, float | bool | int | None]) -> float:
    return weighted_average_normalized(
        [
            (float(feature_map.get("evidence_count", 0.0)), 0.28),
            (float(feature_map.get("specificity_score", 0.0)), 0.24),
            (float(feature_map.get("evidence_richness", 0.0)), 0.20),
            (float(feature_map.get("consistency_score", 0.0)), 0.16),
            (float(feature_map.get("completeness_score", 0.0)), 0.12),
        ]
    )


def _trajectory_raw(
    *,
    feature_map: dict[str, float | bool | int | None],
    semantic_scores: dict[str, float | int],
) -> float:
    return weighted_average_normalized(
        [
            (float(feature_map.get("trajectory_challenge_score", 0.0)), 0.18),
            (float(feature_map.get("trajectory_adaptation_score", 0.0)), 0.24),
            (float(feature_map.get("trajectory_reflection_score", 0.0)), 0.22),
            (float(feature_map.get("trajectory_outcome_score", 0.0)), 0.08),
            (float(feature_map.get("growth_trajectory", 0.0)), 0.10),
            (float(feature_map.get("resilience", 0.0)), 0.08),
            (float(semantic_scores.get("growth_trajectory", 0.0)) / 100.0, 0.07),
            (float(semantic_scores.get("motivation_authenticity", 0.0)) / 100.0, 0.03),
        ]
    )


def build_shortlist_signals(
    *,
    feature_map: dict[str, float | bool | int | None],
    semantic_scores: dict[str, float | int],
    merit_score: int,
    confidence_score: int,
    authenticity_risk: int,
    recommendation: Recommendation | str,
) -> ShortlistSignals:
    evidence_coverage_raw = _evidence_coverage_raw(feature_map)
    trajectory_raw = _trajectory_raw(feature_map=feature_map, semantic_scores=semantic_scores)
    hidden_potential_raw = _hidden_potential_raw(
        feature_map=feature_map,
        semantic_scores=semantic_scores,
        authenticity_risk=authenticity_risk,
    )
    support_needed_raw = _support_needed_raw(
        feature_map=feature_map,
        merit_score=merit_score,
        confidence_score=confidence_score,
    )
    shortlist_priority_raw = weighted_average_normalized(
        [
            (merit_score / 100.0, 0.38),
            (trajectory_raw, 0.18),
            (hidden_potential_raw, 0.18),
            (evidence_coverage_raw, 0.12),
            (confidence_score / 100.0, 0.08),
            (1.0 if recommendation == Recommendation.REVIEW_PRIORITY else 0.0, 0.06),
        ]
    )
    shortlist_priority_raw -= max(0.0, (authenticity_risk / 100.0) - 0.70) * 0.22
    shortlist_priority_raw = clamp01(shortlist_priority_raw)

    return ShortlistSignals(
        hidden_potential_score=to_display_score(hidden_potential_raw),
        support_needed_score=to_display_score(support_needed_raw),
        shortlist_priority_score=to_display_score(shortlist_priority_raw),
        evidence_coverage_score=to_display_score(evidence_coverage_raw),
        trajectory_score=to_display_score(trajectory_raw),
    )


def build_batch_shortlist_summary(results: list[object]) -> BatchShortlistSummary:
    sorted_results = _pairwise_rank_results(results)
    ranked_candidate_ids = [item.candidate_id for item in sorted_results]
    shortlist_candidate_ids = [item.candidate_id for item in sorted_results[: min(5, len(sorted_results))]]
    hidden_potential_candidate_ids = [
        item.candidate_id for item in sorted_results if getattr(item, "hidden_potential_score", 0) >= 25
    ][: min(5, len(sorted_results))]
    support_needed_candidate_ids = [
        item.candidate_id for item in sorted_results if getattr(item, "support_needed_score", 0) >= 55
    ][: min(5, len(sorted_results))]
    authenticity_review_candidate_ids = [
        item.candidate_id for item in sorted_results if getattr(item, "authenticity_risk", 0) >= 45
    ][: min(5, len(sorted_results))]

    return BatchShortlistSummary(
        ranked_candidate_ids=ranked_candidate_ids,
        shortlist_candidate_ids=shortlist_candidate_ids,
        hidden_potential_candidate_ids=hidden_potential_candidate_ids,
        support_needed_candidate_ids=support_needed_candidate_ids,
        authenticity_review_candidate_ids=authenticity_review_candidate_ids,
    )
