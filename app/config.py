"""Global scoring configuration for the MVP service."""

from __future__ import annotations

from dataclasses import dataclass, field


SCORING_VERSION = "v1.0.0"
PROMPT_VERSION: str | None = None

EXCLUDED_FIELDS = [
    "first_name",
    "last_name",
    "middle_name",
    "full_name",
    "iin",
    "id_number",
    "address",
    "phone",
    "email",
    "social_links",
    "family_details",
    "income",
    "social_background",
    "gender",
    "sex",
    "citizenship",
    "ethnicity",
    "race",
    "religion",
]


@dataclass(slots=True)
class ScoringWeights:
    """Weights for merit, confidence, and risk calculations."""

    merit_breakdown: dict[str, float] = field(
        default_factory=lambda: {
            "potential": 0.30,
            "motivation": 0.25,
            "leadership_agency": 0.20,
            "experience_skills": 0.15,
            "trust_completeness": 0.10,
        }
    )
    confidence_components: dict[str, float] = field(
        default_factory=lambda: {
            "specificity_score": 0.30,
            "evidence_count": 0.25,
            "consistency_score": 0.25,
            "completeness_score": 0.20,
        }
    )


@dataclass(slots=True)
class Thresholds:
    """Operational thresholds for recommendation routing."""

    min_words_meaningful_text: int = 40
    min_non_empty_sources: int = 1

    very_low_confidence: float = 0.35
    acceptable_confidence: float = 0.55

    high_merit: float = 0.72
    medium_merit: float = 0.55

    elevated_risk: float = 0.60
    high_risk: float = 0.75

    low_evidence: float = 0.35


@dataclass(slots=True)
class NormalizationConfig:
    """Scales and defaults for structured feature normalization."""

    english_scale_max: dict[str, float] = field(
        default_factory=lambda: {
            "ielts": 9.0,
            "toefl": 120.0,
            "toefl ibt": 120.0,
            "toefl pbt": 677.0,
            "cefr": 6.0,
        }
    )
    certificate_scale_max: dict[str, float] = field(
        default_factory=lambda: {
            "unt": 140.0,
            "nis graduation": 100.0,
            "kazakhstan high school diploma": 100.0,
            "kazakhstan school completion": 100.0,
        }
    )
    unknown_scale_default: float = 0.5


@dataclass(slots=True)
class AppConfig:
    """Application configuration container."""

    scoring_version: str = SCORING_VERSION
    prompt_version: str | None = PROMPT_VERSION
    excluded_fields: list[str] = field(default_factory=lambda: EXCLUDED_FIELDS.copy())
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    thresholds: Thresholds = field(default_factory=Thresholds)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)


CONFIG = AppConfig()
