"""Authenticity risk estimation.

The risk score is a review-risk/uncertainty signal, not proof of cheating and
not an auto-rejection decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.utils.math_utils import clamp01


@dataclass(slots=True)
class AuthenticityResult:
    authenticity_risk_raw: float
    review_flags: list[str]


def estimate_authenticity_risk(features: dict[str, float | bool], diagnostics: dict[str, float | bool | int]) -> AuthenticityResult:
    """Compute authenticity risk from transparent heuristic components."""
    genericness = float(features.get("genericness_score", 0.0))
    evidence_count = float(features.get("evidence_count", 0.0))
    consistency = float(features.get("consistency_score", 0.0))
    polished_but_empty_score = float(features.get("polished_but_empty_score", 0.0))
    cross_section_mismatch_score = float(features.get("cross_section_mismatch_score", 0.0))
    contradiction_flag = bool(features.get("contradiction_flag", False))

    long_but_thin = bool(diagnostics.get("long_but_thin", False))

    polished_but_empty_pattern = long_but_thin and genericness > 0.55 and evidence_count < 0.35

    risk = 0.0
    risk += genericness * 0.35
    risk += (1.0 - evidence_count) * 0.25
    risk += (1.0 - consistency) * 0.18
    risk += polished_but_empty_score * 0.12
    risk += cross_section_mismatch_score * 0.10
    risk += 0.12 if polished_but_empty_pattern else 0.0
    risk += 0.15 if contradiction_flag else 0.0

    # Reduce risk when strong grounded evidence is present.
    if evidence_count > 0.65 and consistency > 0.65:
        risk -= 0.12

    risk = clamp01(risk)

    flags: list[str] = []
    if polished_but_empty_pattern:
        flags.append("polished_but_empty_pattern")
    if polished_but_empty_score > 0.60:
        flags.append("high_polished_but_empty")
    if genericness > 0.60:
        flags.append("high_genericness")
    if cross_section_mismatch_score > 0.55:
        flags.append("cross_section_mismatch")
    if evidence_count < 0.35:
        flags.append("low_evidence_density")
    if consistency < 0.45:
        flags.append("section_mismatch")
    if contradiction_flag:
        flags.append("contradiction_risk")
    if 0.45 <= risk < 0.70:
        flags.append("moderate_authenticity_risk")
    if risk >= 0.70:
        flags.append("high_authenticity_risk")

    return AuthenticityResult(authenticity_risk_raw=risk, review_flags=flags)
