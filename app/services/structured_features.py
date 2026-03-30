"""Structured and process-level feature extraction."""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import CONFIG
from app.services.preprocessing import NormalizedTextBundle
from app.utils.math_utils import clamp01, safe_div


@dataclass(slots=True)
class StructuredFeaturesResult:
    features: dict[str, float | bool]
    notes: list[str]


PROJECT_TERMS = ["project", "club", "initiative", "event", "startup", "volunteer", "research"]
ACHIEVEMENT_TERMS = ["won", "award", "improved", "built", "organized", "launched", "led", "created", "achieved"]
LINK_MARKERS = ["for example", "for instance", "because", "result", "outcome", "therefore"]
NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _normalize_by_scale(score_type: str | None, score: float | None, mapping: dict[str, float], neutral: float) -> tuple[float, str]:
    if score is None:
        return neutral, "score_missing"

    if score_type:
        key = score_type.strip().lower()
    else:
        key = ""

    if key in mapping:
        max_value = mapping[key]
        return clamp01(safe_div(float(score), max_value, default=neutral)), "scale_mapped"

    # Fallback: treat 0..1 as already normalized, 0..100 as percentage-like.
    if 0.0 <= float(score) <= 1.0:
        return clamp01(float(score)), "already_normalized"
    if 1.0 < float(score) <= 100.0:
        return clamp01(float(score) / 100.0), "unknown_scale_percentage_assumption"

    return neutral, "unknown_scale_neutral_default"


def extract_structured_features(
    structured_data: dict[str, object] | None,
    behavioral_signals: dict[str, object] | None,
    bundle: NormalizedTextBundle,
) -> StructuredFeaturesResult:
    """Extract transparent baseline structured/process features in [0..1]."""
    notes: list[str] = []

    education = ((structured_data or {}).get("education") or {}) if isinstance(structured_data, dict) else {}
    english = (education.get("english_proficiency") or {}) if isinstance(education, dict) else {}
    certificate = (education.get("school_certificate") or {}) if isinstance(education, dict) else {}

    english_norm, english_note = _normalize_by_scale(
        score_type=english.get("type") if isinstance(english, dict) else None,
        score=english.get("score") if isinstance(english, dict) else None,
        mapping=CONFIG.normalization.english_scale_max,
        neutral=CONFIG.normalization.unknown_scale_default,
    )
    notes.append(f"english_score:{english_note}")

    cert_norm, cert_note = _normalize_by_scale(
        score_type=certificate.get("type") if isinstance(certificate, dict) else None,
        score=certificate.get("score") if isinstance(certificate, dict) else None,
        mapping=CONFIG.normalization.certificate_scale_max,
        neutral=CONFIG.normalization.unknown_scale_default,
    )
    notes.append(f"certificate_score:{cert_note}")

    logical_groups_present = int(bundle.stats.get("logical_source_groups_present", 0))
    logical_groups_total = max(int(bundle.stats.get("logical_source_groups_total", 3)), 1)
    text_completeness_score = clamp01(logical_groups_present / logical_groups_total)

    total_questions = len(bundle.qa_questions_original)
    answered_questions = len(bundle.motivation_answers_original)
    question_coverage_score = clamp01(safe_div(answered_questions, total_questions, default=1.0 if total_questions == 0 else 0.0))

    behavioral = behavioral_signals or {}
    completion_rate = behavioral.get("completion_rate")
    scenario_depth = behavioral.get("scenario_depth")

    behavioral_completion_score = clamp01(
        (
            (float(completion_rate) if completion_rate is not None else text_completeness_score)
            + (float(scenario_depth) if scenario_depth is not None else text_completeness_score)
        )
        / 2.0
    )

    returned_to_edit_flag = bool(behavioral.get("returned_to_edit") is True)
    skipped_optional = int(behavioral.get("skipped_optional_questions") or 0)
    skipped_optional_questions_penalty = clamp01(min(skipped_optional, 10) / 10.0)

    application_materials = (
        structured_data.get("application_materials")
        if isinstance(structured_data, dict) and isinstance(structured_data.get("application_materials"), dict)
        else {}
    )

    documents = application_materials.get("documents") if isinstance(application_materials.get("documents"), list) else []
    attachments = application_materials.get("attachments") if isinstance(application_materials.get("attachments"), list) else []
    portfolio_links = (
        application_materials.get("portfolio_links")
        if isinstance(application_materials.get("portfolio_links"), list)
        else []
    )
    video_link = (
        application_materials.get("video_presentation_link")
        or application_materials.get("videoPresentationLink")
        or application_materials.get("video_url")
    )

    docs_count = len(documents) + len(attachments)
    docs_count_score = clamp01(docs_count / 6.0)
    portfolio_links_count = len(portfolio_links)
    portfolio_links_score = clamp01(portfolio_links_count / 4.0)
    has_video_presentation = bool(isinstance(video_link, str) and video_link.strip())

    text_lower = bundle.full_text_lower
    tokens = text_lower.split()
    token_count = max(len(tokens), 1)

    evidence_count_estimate = clamp01((len(NUMBER_RE.findall(text_lower)) + sum(text_lower.count(marker) for marker in LINK_MARKERS)) / 20.0)
    linked_examples_count = clamp01(sum(text_lower.count(marker) for marker in LINK_MARKERS) / 15.0)
    achievement_mentions_count = clamp01(sum(text_lower.count(term) for term in ACHIEVEMENT_TERMS) / 20.0)
    project_mentions_count = clamp01(sum(text_lower.count(term) for term in PROJECT_TERMS) / 20.0)

    # Supports confidence calibration from verbosity profile.
    density_norm = clamp01(token_count / 1200.0)

    return StructuredFeaturesResult(
        features={
            "english_score_normalized": english_norm,
            "certificate_score_normalized": cert_norm,
            "text_completeness_score": text_completeness_score,
            "question_coverage_score": question_coverage_score,
            "behavioral_completion_score": behavioral_completion_score,
            "returned_to_edit_flag": returned_to_edit_flag,
            "skipped_optional_questions_penalty": skipped_optional_questions_penalty,
            "docs_count_score": docs_count_score,
            "portfolio_links_score": portfolio_links_score,
            "has_video_presentation": has_video_presentation,
            "evidence_count_estimate": evidence_count_estimate,
            "linked_examples_count": linked_examples_count,
            "achievement_mentions_count": achievement_mentions_count,
            "project_mentions_count": project_mentions_count,
            "verbosity_density_score": density_norm,
        },
        notes=notes,
    )
