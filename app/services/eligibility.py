"""Rule-based eligibility gate.

Eligibility is a formal completeness/admissibility status and must not be
misused as an estimate of candidate merit.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.config import CONFIG
from app.services.preprocessing import NormalizedTextBundle


@dataclass(slots=True)
class EligibilityResult:
    status: str
    reasons: list[str]


def evaluate_eligibility(candidate_id: str | None, consent: bool | None, bundle: NormalizedTextBundle) -> EligibilityResult:
    """Evaluate whether application has enough usable data for scoring."""
    reasons: list[str] = []

    if consent is False:
        return EligibilityResult(status="invalid", reasons=["consent_not_granted"])

    if not candidate_id or not candidate_id.strip():
        return EligibilityResult(status="invalid", reasons=["missing_candidate_id"])

    word_count = int(bundle.stats.get("word_count", 0))
    non_empty_sources = int(bundle.stats.get("non_empty_text_sources", 0))

    if word_count == 0 or non_empty_sources == 0:
        return EligibilityResult(status="invalid", reasons=["no_usable_text_content"])

    if word_count < CONFIG.thresholds.min_words_meaningful_text:
        reasons.append("text_too_short_for_reliable_scoring")

    if non_empty_sources < CONFIG.thresholds.min_non_empty_sources:
        reasons.append("insufficient_text_sources")

    missing_count = sum(int(v) for v in bundle.missingness_map.values())

    if word_count < 20:
        return EligibilityResult(status="incomplete_application", reasons=reasons or ["critical_text_missing"])

    if missing_count >= 2 or reasons:
        return EligibilityResult(status="conditionally_eligible", reasons=reasons or ["high_missingness"])

    return EligibilityResult(status="eligible", reasons=[])
