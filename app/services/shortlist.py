"""Transparent shortlist-oriented derived signals for committee workflow."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.policy import build_policy_snapshot
from app.services.offline_ranker import rank_results_with_offline_ranker
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


@dataclass(slots=True)
class HiddenPotentialDiagnostics:
    underlying_signal: float
    trajectory_signal: float
    self_presentation: float
    overstatement_risk: float
    evidence_floor: float
    credible_action_signal: float
    understatement_gap: float
    modest_presentation: float


def _practical_action_raw(feature_map: dict[str, float | bool | int | None]) -> float:
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


def _hidden_potential_diagnostics(
    *,
    feature_map: dict[str, float | bool | int | None],
    semantic_scores: dict[str, float | int],
) -> HiddenPotentialDiagnostics:
    underlying_signal = _underlying_signal_raw(feature_map=feature_map, semantic_scores=semantic_scores)
    trajectory_signal = _trajectory_raw(feature_map=feature_map, semantic_scores=semantic_scores)
    self_presentation = _self_presentation_raw(feature_map=feature_map, semantic_scores=semantic_scores)
    practical_action_signal = _practical_action_raw(feature_map)
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
            (practical_action_signal, 0.42),
            (float(feature_map.get("trajectory_adaptation_score", 0.0)), 0.18),
            (float(feature_map.get("trajectory_reflection_score", 0.0)), 0.16),
            (float(feature_map.get("trajectory_outcome_score", 0.0)), 0.12),
            (float(feature_map.get("evidence_count", 0.0)), 0.12),
        ]
    )
    understatement_gap = clamp01(underlying_signal - self_presentation)
    modest_presentation = clamp01(0.62 - self_presentation)
    return HiddenPotentialDiagnostics(
        underlying_signal=underlying_signal,
        trajectory_signal=trajectory_signal,
        self_presentation=self_presentation,
        overstatement_risk=overstatement_risk,
        evidence_floor=evidence_floor,
        credible_action_signal=credible_action_signal,
        understatement_gap=understatement_gap,
        modest_presentation=modest_presentation,
    )


def _hidden_potential_raw(
    *,
    feature_map: dict[str, float | bool | int | None],
    semantic_scores: dict[str, float | int],
    authenticity_risk: int,
) -> float:
    diagnostics = _hidden_potential_diagnostics(feature_map=feature_map, semantic_scores=semantic_scores)
    low_evidence_penalty = max(0.0, 0.22 - float(feature_map.get("evidence_count", 0.0)))
    inflated_presentation_penalty = 0.0
    if diagnostics.self_presentation >= 0.58 and diagnostics.overstatement_risk >= 0.48:
        inflated_presentation_penalty += 0.10
    if diagnostics.understatement_gap <= 0.08:
        inflated_presentation_penalty += 0.07
    if diagnostics.credible_action_signal < 0.16 and diagnostics.self_presentation >= 0.52:
        inflated_presentation_penalty += 0.10
    if (
        diagnostics.evidence_floor < 0.32
        and diagnostics.overstatement_risk >= 0.45
        and diagnostics.credible_action_signal < 0.18
    ):
        inflated_presentation_penalty += 0.08

    raw = (
        (diagnostics.underlying_signal * 0.30)
        + (diagnostics.trajectory_signal * 0.14)
        + (diagnostics.credible_action_signal * 0.24)
        + (diagnostics.understatement_gap * 0.14)
        + (diagnostics.evidence_floor * 0.14)
        + (diagnostics.modest_presentation * 0.04)
    )
    raw += min(0.06, (diagnostics.credible_action_signal * 0.10) + (diagnostics.trajectory_signal * 0.06))
    raw -= diagnostics.overstatement_risk * 0.08
    raw -= low_evidence_penalty * 0.20
    raw -= inflated_presentation_penalty
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
    sorted_results = rank_results_with_offline_ranker(results)
    ranked_candidate_ids = [item.candidate_id for item in sorted_results]
    shortlist_candidate_ids = [
        item.candidate_id
        for item in sorted_results
        if build_policy_snapshot(
            merit_score=int(getattr(item, "merit_score", 0)),
            confidence_score=int(getattr(item, "confidence_score", 0)),
            authenticity_risk=int(getattr(item, "authenticity_risk", 0)),
            hidden_potential_score=int(getattr(item, "hidden_potential_score", 0)),
            support_needed_score=int(getattr(item, "support_needed_score", 0)),
            shortlist_priority_score=int(getattr(item, "shortlist_priority_score", 0)),
            evidence_coverage_score=int(getattr(item, "evidence_coverage_score", 0)),
            trajectory_score=int(getattr(item, "trajectory_score", 0)),
        ).shortlist_band
    ][: min(5, len(sorted_results))]
    hidden_potential_candidate_ids = [
        item.candidate_id
        for item in sorted_results
        if build_policy_snapshot(
            merit_score=int(getattr(item, "merit_score", 0)),
            confidence_score=int(getattr(item, "confidence_score", 0)),
            authenticity_risk=int(getattr(item, "authenticity_risk", 0)),
            hidden_potential_score=int(getattr(item, "hidden_potential_score", 0)),
            support_needed_score=int(getattr(item, "support_needed_score", 0)),
            shortlist_priority_score=int(getattr(item, "shortlist_priority_score", 0)),
            evidence_coverage_score=int(getattr(item, "evidence_coverage_score", 0)),
            trajectory_score=int(getattr(item, "trajectory_score", 0)),
        ).hidden_potential_band
    ][: min(5, len(sorted_results))]
    support_needed_candidate_ids = [
        item.candidate_id
        for item in sorted_results
        if build_policy_snapshot(
            merit_score=int(getattr(item, "merit_score", 0)),
            confidence_score=int(getattr(item, "confidence_score", 0)),
            authenticity_risk=int(getattr(item, "authenticity_risk", 0)),
            hidden_potential_score=int(getattr(item, "hidden_potential_score", 0)),
            support_needed_score=int(getattr(item, "support_needed_score", 0)),
            shortlist_priority_score=int(getattr(item, "shortlist_priority_score", 0)),
            evidence_coverage_score=int(getattr(item, "evidence_coverage_score", 0)),
            trajectory_score=int(getattr(item, "trajectory_score", 0)),
        ).support_needed_band
    ][: min(5, len(sorted_results))]
    authenticity_review_candidate_ids = [
        item.candidate_id
        for item in sorted_results
        if build_policy_snapshot(
            merit_score=int(getattr(item, "merit_score", 0)),
            confidence_score=int(getattr(item, "confidence_score", 0)),
            authenticity_risk=int(getattr(item, "authenticity_risk", 0)),
            hidden_potential_score=int(getattr(item, "hidden_potential_score", 0)),
            support_needed_score=int(getattr(item, "support_needed_score", 0)),
            shortlist_priority_score=int(getattr(item, "shortlist_priority_score", 0)),
            evidence_coverage_score=int(getattr(item, "evidence_coverage_score", 0)),
            trajectory_score=int(getattr(item, "trajectory_score", 0)),
        ).authenticity_review_band
    ][: min(5, len(sorted_results))]

    return BatchShortlistSummary(
        ranked_candidate_ids=ranked_candidate_ids,
        shortlist_candidate_ids=shortlist_candidate_ids,
        hidden_potential_candidate_ids=hidden_potential_candidate_ids,
        support_needed_candidate_ids=support_needed_candidate_ids,
        authenticity_review_candidate_ids=authenticity_review_candidate_ids,
    )
