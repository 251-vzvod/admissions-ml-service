"""Heuristic rubric extractor for potential, confidence, risk, and consistency signals."""

from __future__ import annotations

from dataclasses import dataclass
import re

from app.services.preprocessing import NormalizedTextBundle
from app.utils.math_utils import clamp01, safe_div, weighted_average_normalized
from app.utils.text import count_occurrences


TOKEN_RE = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


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
    "initiated",
    "organized",
    "инициировал",
    "организовал",
    "создал",
    "запустил",
    "улучшил",
    "помог",
    "собрал",
    "изменил",
]

ROLE_TERMS = [
    "team",
    "club",
    "leader",
    "captain",
    "mentor",
    "volunteer",
    "group",
    "community",
    "classmates",
    "students",
    "группа",
    "команда",
    "лидер",
    "наставник",
    "волонтер",
    "сообщество",
    "одноклассники",
    "ученики",
]

OUTCOME_TERMS = [
    "result",
    "outcome",
    "improved",
    "increased",
    "saved",
    "won",
    "better",
    "changed",
    "resulted",
    "результат",
    "улучш",
    "добил",
    "смог",
    "измен",
]

TEMPORAL_TERMS = [
    "last year",
    "during",
    "after",
    "before",
    "for months",
    "for weeks",
    "later",
    "at first",
    "then",
    "when",
    "когда",
    "после",
    "до",
    "сначала",
    "потом",
    "позже",
    "несколько месяцев",
]

CAUSE_EFFECT_TERMS = [
    "because",
    "therefore",
    "so that",
    "as a result",
    "because of",
    "which meant",
    "so i",
    "потому",
    "поэтому",
    "в результате",
    "чтобы",
]

RESILIENCE_TERMS = [
    "difficult",
    "challenge",
    "failed",
    "setback",
    "tried again",
    "hard",
    "uncertainty",
    "struggle",
    "сложно",
    "трудно",
    "не сдался",
    "ошиб",
    "не получилось",
    "тяжело",
    "трудност",
]

ADAPTATION_TERMS = [
    "changed",
    "adjusted",
    "adapted",
    "rewrote",
    "revised",
    "improved",
    "then i changed",
    "after that i",
    "learned how",
    "feedback",
    "изменил",
    "переделал",
    "адаптировал",
    "улучшил",
    "обратная связь",
    "скорректировал",
]

REFLECTION_TERMS = [
    "i realized",
    "i learned",
    "i understood",
    "it made me",
    "now i think",
    "after that",
    "i became",
    "what changed in me",
    "i noticed",
    "я понял",
    "я осознал",
    "я научился",
    "это изменило меня",
    "я заметил",
    "я стал",
]

PROGRAM_FIT_TERMS = [
    "invision",
    "program",
    "interdisciplinary",
    "teamwork",
    "projects",
    "mission",
    "university",
    "community",
    "education",
    "program-based",
    "программа",
    "университет",
    "проект",
    "команд",
    "мисси",
    "обучени",
    "сообщество",
]

GENERIC_TERMS = [
    "i am passionate",
    "i am motivated",
    "i want to grow",
    "dream",
    "best version",
    "change the world",
    "unlock my potential",
    "meaningful impact",
    "я мотивирован",
    "я очень мотивирован",
    "хочу развиваться",
    "мечтаю",
    "изменить мир",
]

NUMERIC_TOKENS = [str(i) for i in range(10)]

CONTRADICTION_PAIRS = [
    ("i never", "i always"),
    ("no experience", "i led"),
    ("do not know", "clear plan"),
    ("никогда", "всегда"),
    ("нет опыта", "я руководил"),
]

STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "that",
    "this",
    "from",
    "into",
    "when",
    "what",
    "have",
    "just",
    "they",
    "them",
    "their",
    "your",
    "about",
    "after",
    "before",
    "because",
    "then",
    "were",
    "was",
    "had",
    "you",
    "our",
    "but",
    "или",
    "как",
    "что",
    "это",
    "когда",
    "потом",
    "после",
    "перед",
    "для",
    "его",
    "ее",
    "они",
    "она",
    "оно",
    "мне",
    "меня",
    "свои",
    "свою",
    "своих",
}


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


def _content_tokens(text: str) -> set[str]:
    lowered = text.lower()
    tokens = set()
    for token in TOKEN_RE.findall(lowered):
        if len(token) < 3:
            continue
        if token in STOPWORDS:
            continue
        tokens.add(token)
    return tokens


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def extract_text_features(bundle: NormalizedTextBundle, structured: dict[str, float | bool]) -> TextFeaturesResult:
    """Extract heuristic rubric features in normalized [0..1] scale."""
    full_text = bundle.full_text_lower
    word_count = int(bundle.stats.get("word_count", 0))

    first_person_density = clamp01(
        (
            full_text.count(" i ")
            + full_text.count(" my ")
            + full_text.count(" me ")
            + full_text.count(" я ")
            + full_text.count(" мой ")
            + full_text.count(" меня ")
        )
        / 40.0
    )
    action_density = _density_score(full_text, ACTION_TERMS, denominator=24.0)
    role_density = _density_score(full_text, ROLE_TERMS, denominator=16.0)
    outcome_density = _density_score(full_text, OUTCOME_TERMS, denominator=16.0)
    temporal_density = _density_score(full_text, TEMPORAL_TERMS, denominator=16.0)
    causal_density = _density_score(full_text, CAUSE_EFFECT_TERMS, denominator=20.0)
    resilience_density = _density_score(full_text, RESILIENCE_TERMS, denominator=16.0)
    adaptation_density = _density_score(full_text, ADAPTATION_TERMS, denominator=12.0)
    reflection_density = _density_score(full_text, REFLECTION_TERMS, denominator=12.0)
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
            (float(structured.get("linked_examples_count", 0.0)), 0.15),
        ]
    )
    leadership_impact = weighted_average_normalized(
        [
            (role_density, 0.30),
            (outcome_density, 0.35),
            (float(structured.get("project_mentions_count", 0.0)), 0.20),
            (number_density, 0.15),
        ]
    )
    growth_trajectory = weighted_average_normalized(
        [
            (temporal_density, 0.22),
            (resilience_density, 0.24),
            (adaptation_density, 0.22),
            (reflection_density, 0.18),
            (causal_density, 0.10),
            (first_person_density, 0.04),
        ]
    )
    resilience = weighted_average_normalized(
        [
            (resilience_density, 0.40),
            (adaptation_density, 0.15),
            (temporal_density, 0.20),
            (evidence_density, 0.15),
            (first_person_density, 0.10),
        ]
    )
    program_fit = weighted_average_normalized(
        [
            (program_fit_density, 0.60),
            (motivation_clarity, 0.25),
            (float(structured.get("question_coverage_score", 0.0)), 0.15),
        ]
    )
    evidence_richness = weighted_average_normalized(
        [
            (evidence_density, 0.55),
            (float(structured.get("evidence_count_estimate", 0.0)), 0.20),
            (number_density, 0.15),
            (float(structured.get("linked_examples_count", 0.0)), 0.10),
        ]
    )

    specificity_score = weighted_average_normalized(
        [
            (evidence_density, 0.45),
            (number_density, 0.25),
            (float(structured.get("linked_examples_count", 0.20)), 0.20),
            (float(structured.get("achievement_mentions_count", 0.10)), 0.10),
        ]
    )
    evidence_count = weighted_average_normalized(
        [
            (float(structured.get("evidence_count_estimate", 0.0)), 0.5),
            (float(structured.get("linked_examples_count", 0.0)), 0.3),
            (evidence_density, 0.2),
        ]
    )
    trajectory_challenge_score = weighted_average_normalized(
        [
            (resilience_density, 0.45),
            (temporal_density, 0.20),
            (causal_density, 0.15),
            (first_person_density, 0.20),
        ]
    )
    trajectory_adaptation_score = weighted_average_normalized(
        [
            (adaptation_density, 0.45),
            (causal_density, 0.20),
            (temporal_density, 0.20),
            (action_density, 0.15),
        ]
    )
    trajectory_reflection_score = weighted_average_normalized(
        [
            (reflection_density, 0.50),
            (causal_density, 0.20),
            (temporal_density, 0.15),
            (first_person_density, 0.15),
        ]
    )
    trajectory_outcome_score = weighted_average_normalized(
        [
            (outcome_density, 0.45),
            (number_density, 0.25),
            (evidence_density, 0.20),
            (action_density, 0.10),
        ]
    )

    contradiction_flag = _flag_contradictions(full_text)

    interview_family_present = (
        int(not bundle.missingness_map.get("interview_text", True))
        or int(not bundle.missingness_map.get("video_interview_transcript_text", True))
        or int(not bundle.missingness_map.get("video_presentation_transcript_text", True))
    )
    section_presence = [
        int(not bundle.missingness_map.get("motivation_letter_text", True)),
        int(not bundle.missingness_map.get("motivation_questions", True)),
        int(bool(interview_family_present)),
    ]
    section_alignment = clamp01(sum(section_presence) / 3.0)

    section_texts = [
        bundle.motivation_letter_original,
        interview_text := bundle.interview_original,
        bundle.video_interview_transcript_original,
        bundle.video_presentation_transcript_original,
        "\n".join(bundle.motivation_answers_original),
    ]
    non_empty_sections = [text for text in section_texts if text.strip()]
    section_pairs = [_section_metric(text) for text in non_empty_sections]

    lexical_overlaps: list[float] = []
    role_gaps: list[float] = []
    temporal_gaps: list[float] = []
    evidence_gaps: list[float] = []
    generic_gaps: list[float] = []

    for i in range(len(non_empty_sections)):
        left_text = non_empty_sections[i].lower()
        left_tokens = _content_tokens(non_empty_sections[i])
        left_role_density = _density_score(left_text, ROLE_TERMS, denominator=8.0)
        left_temporal_density = _density_score(left_text, TEMPORAL_TERMS, denominator=8.0)
        for j in range(i + 1, len(non_empty_sections)):
            right_text = non_empty_sections[j].lower()
            right_tokens = _content_tokens(non_empty_sections[j])
            right_role_density = _density_score(right_text, ROLE_TERMS, denominator=8.0)
            right_temporal_density = _density_score(right_text, TEMPORAL_TERMS, denominator=8.0)

            lexical_overlaps.append(_jaccard(left_tokens, right_tokens))
            role_gaps.append(abs(left_role_density - right_role_density))
            temporal_gaps.append(abs(left_temporal_density - right_temporal_density))
            evidence_gaps.append(abs(section_pairs[i][0] - section_pairs[j][0]))
            generic_gaps.append(abs(section_pairs[i][1] - section_pairs[j][1]))

    section_claim_overlap_score = clamp01(
        safe_div(sum(lexical_overlaps), len(lexical_overlaps), default=0.0) if lexical_overlaps else 0.0
    )
    section_role_consistency_score = clamp01(
        1.0 - safe_div(sum(role_gaps), len(role_gaps), default=0.0) if role_gaps else 1.0
    )
    section_time_consistency_score = clamp01(
        1.0 - safe_div(sum(temporal_gaps), len(temporal_gaps), default=0.0) if temporal_gaps else 1.0
    )

    cross_section_mismatch_score = clamp01(
        weighted_average_normalized(
            [
                (safe_div(sum(evidence_gaps), len(evidence_gaps), default=0.0), 0.28),
                (safe_div(sum(generic_gaps), len(generic_gaps), default=0.0), 0.18),
                (1.0 - section_claim_overlap_score, 0.28),
                (1.0 - section_role_consistency_score, 0.16),
                (1.0 - section_time_consistency_score, 0.10),
            ],
            default=0.0,
        )
    )

    consistency_score = weighted_average_normalized(
        [
            (section_alignment, 0.22),
            (1.0 - abs(motivation_clarity - program_fit), 0.18),
            (1.0 - abs(initiative - leadership_impact), 0.16),
            (section_claim_overlap_score, 0.18),
            (section_role_consistency_score, 0.14),
            (section_time_consistency_score, 0.12),
        ]
    )
    if contradiction_flag:
        consistency_score = clamp01(consistency_score - 0.25)

    completeness_score = weighted_average_normalized(
        [
            (float(structured.get("text_completeness_score", 0.0)), 0.45),
            (float(structured.get("question_coverage_score", 0.0)), 0.30),
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
        + (0.10 if long_but_thin else 0.0)
    )

    polished_but_empty_score = clamp01(
        weighted_average_normalized(
            [
                (generic_density, 0.38),
                (1.0 - evidence_density, 0.28),
                (1.0 - specificity_score, 0.18),
                (1.0 - section_claim_overlap_score, 0.16),
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
            "trajectory_challenge_score": trajectory_challenge_score,
            "trajectory_adaptation_score": trajectory_adaptation_score,
            "trajectory_reflection_score": trajectory_reflection_score,
            "trajectory_outcome_score": trajectory_outcome_score,
            "section_claim_overlap_score": section_claim_overlap_score,
            "section_role_consistency_score": section_role_consistency_score,
            "section_time_consistency_score": section_time_consistency_score,
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
            "section_pair_count": len(lexical_overlaps),
        },
    )
