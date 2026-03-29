"""Heuristic rubric extractor for potential, confidence, and risk signals."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.preprocessing import NormalizedTextBundle
from app.utils.math_utils import clamp01, safe_div, weighted_average_normalized
from app.utils.text import count_occurrences


ACTION_TERMS = [
    "built",
    "organized",
    "led",
    "started",
    "created",
    "launched",
    "improved",
    "implemented",
    "solved",
    "helped",
    "инициировал",
    "организовал",
    "создал",
    "запустил",
    "помог",
]

ROLE_TERMS = ["team", "club", "leader", "captain", "mentor", "volunteer", "группа", "команда", "лидер"]
OUTCOME_TERMS = ["result", "outcome", "improved", "increased", "saved", "won", "better", "результат", "улучш"]
TEMPORAL_TERMS = ["last year", "during", "after", "before", "for months", "2 years", "когда", "после", "до"]
CAUSE_EFFECT_TERMS = ["because", "therefore", "so that", "as a result", "because of", "потому", "поэтому", "в результате"]
RESILIENCE_TERMS = ["difficult", "challenge", "failed", "setback", "tried again", "hard", "сложно", "трудно", "не сдался"]
PROGRAM_FIT_TERMS = ["invision", "program", "interdisciplinary", "teamwork", "projects", "mission", "university"]
GENERIC_TERMS = ["i am passionate", "i am motivated", "i want to grow", "dream", "best version", "change the world"]
NUMERIC_TOKENS = [str(i) for i in range(10)]

CONTRADICTION_PAIRS = [
    ("i never", "i always"),
    ("no experience", "i led"),
    ("do not know", "clear plan"),
]


@dataclass(slots=True)
class TextFeaturesResult:
    features: dict[str, float | bool]
    diagnostics: dict[str, float | bool | int]


def _density_score(text: str, terms: list[str], denominator: float) -> float:
    return clamp01(count_occurrences(text, terms) / denominator)


def _flag_contradictions(text: str) -> bool:
    lowered = text.lower()
    for left, right in CONTRADICTION_PAIRS:
        if left in lowered and right in lowered:
            return True
    return False


def _section_metric(section_text: str) -> tuple[float, float]:
    section_lower = section_text.lower()
    evidence = clamp01(
        weighted_average_normalized(
            [
                (_density_score(section_lower, ACTION_TERMS, denominator=8.0), 0.35),
                (_density_score(section_lower, OUTCOME_TERMS, denominator=8.0), 0.30),
                (_density_score(section_lower, TEMPORAL_TERMS, denominator=10.0), 0.20),
                (_density_score(section_lower, CAUSE_EFFECT_TERMS, denominator=10.0), 0.15),
            ],
            default=0.0,
        )
    )
    generic = _density_score(section_lower, GENERIC_TERMS, denominator=6.0)
    return evidence, generic


def extract_text_features(bundle: NormalizedTextBundle, structured: dict[str, float | bool]) -> TextFeaturesResult:
    """Extract heuristic rubric features in normalized [0..1] scale."""
    full_text = bundle.full_text_lower
    word_count = int(bundle.stats.get("word_count", 0))

    first_person_density = clamp01(
        (full_text.count(" i ") + full_text.count(" my ") + full_text.count(" me ") + full_text.count(" я ")) / 40.0
    )
    action_density = _density_score(full_text, ACTION_TERMS, denominator=24.0)
    role_density = _density_score(full_text, ROLE_TERMS, denominator=16.0)
    outcome_density = _density_score(full_text, OUTCOME_TERMS, denominator=16.0)
    temporal_density = _density_score(full_text, TEMPORAL_TERMS, denominator=16.0)
    causal_density = _density_score(full_text, CAUSE_EFFECT_TERMS, denominator=20.0)
    resilience_density = _density_score(full_text, RESILIENCE_TERMS, denominator=16.0)
    program_fit_density = _density_score(full_text, PROGRAM_FIT_TERMS, denominator=10.0)

    number_density = clamp01(sum(full_text.count(d) for d in NUMERIC_TOKENS) / 20.0)
    generic_density = _density_score(full_text, GENERIC_TERMS, denominator=10.0)
    evidence_density = clamp01(
        weighted_average_normalized(
            [
                (action_density, 0.30),
                (outcome_density, 0.25),
                (temporal_density, 0.20),
                (causal_density, 0.15),
                (number_density, 0.10),
            ],
            default=0.0,
        )
    )

    motivation_clarity = weighted_average_normalized(
        [
            (program_fit_density, 0.35),
            (causal_density, 0.20),
            (first_person_density, 0.15),
            (evidence_density, 0.30),
        ]
    )
    initiative = weighted_average_normalized(
        [
            (action_density, 0.45),
            (role_density, 0.20),
            (outcome_density, 0.20),
            (structured.get("linked_examples_count", 0.0), 0.15),
        ]
    )
    leadership_impact = weighted_average_normalized(
        [
            (role_density, 0.30),
            (outcome_density, 0.35),
            (structured.get("project_mentions_count", 0.0), 0.20),
            (number_density, 0.15),
        ]
    )
    growth_trajectory = weighted_average_normalized(
        [
            (temporal_density, 0.35),
            (resilience_density, 0.35),
            (causal_density, 0.20),
            (first_person_density, 0.10),
        ]
    )
    resilience = weighted_average_normalized(
        [
            (resilience_density, 0.45),
            (temporal_density, 0.25),
            (evidence_density, 0.20),
            (first_person_density, 0.10),
        ]
    )
    program_fit = weighted_average_normalized(
        [
            (program_fit_density, 0.60),
            (motivation_clarity, 0.25),
            (structured.get("question_coverage_score", 0.0), 0.15),
        ]
    )
    evidence_richness = weighted_average_normalized(
        [
            (evidence_density, 0.55),
            (structured.get("evidence_count_estimate", 0.0), 0.20),
            (number_density, 0.15),
            (structured.get("linked_examples_count", 0.0), 0.10),
        ]
    )

    specificity_score = weighted_average_normalized(
        [
            (evidence_density, 0.45),
            (number_density, 0.25),
            (structured.get("linked_examples_count", 0.20), 0.20),
            (structured.get("achievement_mentions_count", 0.10), 0.10),
        ]
    )
    evidence_count = weighted_average_normalized(
        [
            (structured.get("evidence_count_estimate", 0.0), 0.5),
            (structured.get("linked_examples_count", 0.0), 0.3),
            (evidence_density, 0.2),
        ]
    )

    contradiction_flag = _flag_contradictions(full_text)

    section_presence = [
        int(not bundle.missingness_map.get("motivation_letter_text", True)),
        int(not bundle.missingness_map.get("motivation_questions", True)),
        int(not bundle.missingness_map.get("interview_text", True)),
    ]
    section_alignment = clamp01(sum(section_presence) / 3.0)

    consistency_score = weighted_average_normalized(
        [
            (section_alignment, 0.40),
            (1.0 - abs(motivation_clarity - program_fit), 0.30),
            (1.0 - abs(initiative - leadership_impact), 0.30),
        ]
    )
    if contradiction_flag:
        consistency_score = clamp01(consistency_score - 0.25)

    completeness_score = weighted_average_normalized(
        [
            (structured.get("text_completeness_score", 0.0), 0.45),
            (structured.get("question_coverage_score", 0.0), 0.30),
            (clamp01(safe_div(word_count, 500.0)), 0.25),
        ]
    )

    long_but_thin = word_count > 350 and evidence_density < 0.25
    genericness_score = clamp01(
        weighted_average_normalized(
            [
                (generic_density, 0.45),
                (1.0 - evidence_density, 0.35),
                (1.0 - specificity_score, 0.20),
            ]
        )
        + (0.1 if long_but_thin else 0.0)
    )

    section_texts = [bundle.motivation_letter_original, bundle.interview_original, "\n".join(bundle.motivation_answers_original)]
    section_pairs = []
    for text in section_texts:
        if text.strip():
            section_pairs.append(_section_metric(text))

    mismatch_components: list[float] = []
    for i in range(len(section_pairs)):
        for j in range(i + 1, len(section_pairs)):
            mismatch_components.append(abs(section_pairs[i][0] - section_pairs[j][0]))
            mismatch_components.append(abs(section_pairs[i][1] - section_pairs[j][1]))
    cross_section_mismatch_score = clamp01(sum(mismatch_components) / max(len(mismatch_components), 1))

    polished_but_empty_score = clamp01(
        weighted_average_normalized(
            [
                (generic_density, 0.45),
                (1.0 - evidence_density, 0.35),
                (1.0 - specificity_score, 0.20),
            ],
            default=0.0,
        )
        + (0.12 if long_but_thin else 0.0)
    )

    low_evidence_flag = bool(evidence_count < 0.35)

    return TextFeaturesResult(
        features={
            "motivation_clarity": motivation_clarity,
            "initiative": initiative,
            "leadership_impact": leadership_impact,
            "growth_trajectory": growth_trajectory,
            "resilience": resilience,
            "program_fit": program_fit,
            "evidence_richness": evidence_richness,
            "specificity_score": specificity_score,
            "evidence_count": evidence_count,
            "consistency_score": consistency_score,
            "completeness_score": completeness_score,
            "genericness_score": genericness_score,
            "contradiction_flag": contradiction_flag,
            "low_evidence_flag": low_evidence_flag,
            "polished_but_empty_score": polished_but_empty_score,
            "cross_section_mismatch_score": cross_section_mismatch_score,
        },
        diagnostics={
            "word_count": word_count,
            "evidence_density": evidence_density,
            "generic_density": generic_density,
            "long_but_thin": long_but_thin,
        },
    )
