"""Unified routing and shortlist policy thresholds for committee workflow."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import CONFIG


@dataclass(frozen=True, slots=True)
class RoutingPolicySnapshot:
    priority_band: bool
    shortlist_band: bool
    hidden_potential_band: bool
    support_needed_band: bool
    authenticity_review_band: bool
    insufficient_evidence_band: bool


def build_policy_snapshot(
    *,
    merit_score: int,
    confidence_score: int,
    authenticity_risk: int,
    hidden_potential_score: int,
    support_needed_score: int,
    shortlist_priority_score: int,
    evidence_coverage_score: int,
    trajectory_score: int,
) -> RoutingPolicySnapshot:
    cfg = CONFIG.policy

    insufficient_evidence_band = (
        confidence_score <= cfg.insufficient_evidence_confidence_max
        and evidence_coverage_score <= cfg.insufficient_evidence_coverage_max
    )
    authenticity_review_band = authenticity_risk >= cfg.authenticity_review_risk_min
    hidden_potential_band = (
        hidden_potential_score >= cfg.hidden_potential_score_min
        and trajectory_score >= cfg.hidden_potential_trajectory_min
        and evidence_coverage_score >= cfg.hidden_potential_coverage_min
    )
    support_needed_band = (
        support_needed_score >= cfg.support_needed_score_min
        and merit_score >= cfg.support_needed_merit_min
    )
    priority_band = (
        merit_score >= cfg.priority_merit_min
        and confidence_score >= cfg.priority_confidence_min
        and authenticity_risk <= cfg.priority_authenticity_risk_max
        and evidence_coverage_score >= cfg.priority_coverage_min
    )
    shortlist_band = (
        shortlist_priority_score >= cfg.shortlist_priority_min
        and merit_score >= cfg.shortlist_merit_min
        and evidence_coverage_score >= cfg.shortlist_coverage_min
        and authenticity_risk <= cfg.shortlist_authenticity_risk_max
    ) or priority_band or hidden_potential_band

    return RoutingPolicySnapshot(
        priority_band=priority_band,
        shortlist_band=shortlist_band,
        hidden_potential_band=hidden_potential_band,
        support_needed_band=support_needed_band,
        authenticity_review_band=authenticity_review_band,
        insufficient_evidence_band=insufficient_evidence_band,
    )
