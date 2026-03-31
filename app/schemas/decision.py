"""Canonical decision enums used across scoring pipeline and API schema."""

from __future__ import annotations

from enum import StrEnum


class Recommendation(StrEnum):
    INVALID = "invalid"
    INCOMPLETE_APPLICATION = "incomplete_application"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    REVIEW_PRIORITY = "review_priority"
    MANUAL_REVIEW_REQUIRED = "manual_review_required"
    STANDARD_REVIEW = "standard_review"


class ReviewFlag(StrEnum):
    ELIGIBILITY_GATE = "eligibility_gate"
    LOW_CONFIDENCE = "low_confidence"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"
    LOW_EVIDENCE_DENSITY = "low_evidence_density"
    MODERATE_AUTHENTICITY_RISK = "moderate_authenticity_risk"
    HIGH_AUTHENTICITY_RISK = "high_authenticity_risk"
    CONTRADICTION_RISK = "contradiction_risk"
    POSSIBLE_CONTRADICTION = "possible_contradiction"
    POLISHED_BUT_EMPTY_PATTERN = "polished_but_empty_pattern"
    HIGH_POLISHED_BUT_EMPTY = "high_polished_but_empty"
    HIGH_GENERICNESS = "high_genericness"
    CROSS_SECTION_MISMATCH = "cross_section_mismatch"
    SECTION_MISMATCH = "section_mismatch"
    AUXILIARY_AI_GENERATION_SIGNAL = "auxiliary_ai_generation_signal"
    MISSING_REQUIRED_MATERIALS = "missing_required_materials"
