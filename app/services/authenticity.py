"""Authenticity risk estimation.

The risk score is a review-risk/uncertainty signal, not proof of cheating and
not an auto-rejection decision.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.decision import ReviewFlag
from app.services.ai_detector import AIDetectorResult
from app.utils.math_utils import clamp01


@dataclass(slots=True)
class AuthenticityResult:
    authenticity_risk_raw: float
    review_flags: list[ReviewFlag]
    review_reasons: list[str]
    ai_detector_result: AIDetectorResult | None = None


def estimate_authenticity_risk(
    features: dict[str, float | bool],
    diagnostics: dict[str, float | bool | int],
    ai_detector_result: AIDetectorResult | None = None,
) -> AuthenticityResult:
    """Compute authenticity risk from transparent heuristic components."""
    genericness = float(features.get("genericness_score", 0.0))
    evidence_count = float(features.get("evidence_count", 0.0))
    consistency = float(features.get("consistency_score", 0.0))
    polished_but_empty_score = float(features.get("polished_but_empty_score", 0.0))
    cross_section_mismatch_score = float(features.get("cross_section_mismatch_score", 0.0))
    contradiction_flag = bool(features.get("contradiction_flag", False))
    ai_probability = ai_detector_result.probability_ai_generated if ai_detector_result else None

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
    if ai_detector_result and ai_detector_result.applicable and ai_probability is not None:
        ai_penalty = max(0.0, ai_probability - 0.72) * 0.12
        if evidence_count > 0.65 and consistency > 0.65:
            ai_penalty *= 0.5
        risk += ai_penalty

    # Reduce risk when strong grounded evidence is present.
    if evidence_count > 0.65 and consistency > 0.65:
        risk -= 0.12

    risk = clamp01(risk)

    flags: list[ReviewFlag] = []
    reasons: list[str] = []
    if polished_but_empty_pattern:
        flags.append(ReviewFlag.POLISHED_BUT_EMPTY_PATTERN)
        reasons.append("Long and polished text has limited grounded evidence.")
    if polished_but_empty_score > 0.60:
        flags.append(ReviewFlag.HIGH_POLISHED_BUT_EMPTY)
        reasons.append("Text looks polished relative to the amount of concrete evidence provided.")
    if genericness > 0.60:
        flags.append(ReviewFlag.HIGH_GENERICNESS)
        reasons.append("Language is relatively generic and low-specificity.")
    if cross_section_mismatch_score > 0.55:
        flags.append(ReviewFlag.CROSS_SECTION_MISMATCH)
        reasons.append("Different sections vary too much in evidence and groundedness.")
    if evidence_count < 0.35:
        flags.append(ReviewFlag.LOW_EVIDENCE_DENSITY)
        reasons.append("Claims are under-supported by concrete actions, examples, or outcomes.")
    if consistency < 0.45:
        flags.append(ReviewFlag.SECTION_MISMATCH)
        reasons.append("Application sections do not align well enough for high-confidence trust.")
    if contradiction_flag:
        flags.append(ReviewFlag.CONTRADICTION_RISK)
        flags.append(ReviewFlag.POSSIBLE_CONTRADICTION)
        reasons.append("Possible contradictions or unsupported shifts appear across the narrative.")
    if (
        ai_detector_result
        and ai_detector_result.applicable
        and ai_probability is not None
        and ai_probability >= 0.80
    ):
        flags.append(ReviewFlag.AUXILIARY_AI_GENERATION_SIGNAL)
        reasons.append(
            "Auxiliary English-only AI detector assigned elevated AI-likeness; treat this only as a manual review signal."
        )
    if 0.45 <= risk < 0.70:
        flags.append(ReviewFlag.MODERATE_AUTHENTICITY_RISK)
    if risk >= 0.70:
        flags.append(ReviewFlag.HIGH_AUTHENTICITY_RISK)

    return AuthenticityResult(
        authenticity_risk_raw=risk,
        review_flags=flags,
        review_reasons=reasons[:4],
        ai_detector_result=ai_detector_result,
    )
