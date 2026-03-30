"""Rule-based eligibility gate.

Eligibility is a formal completeness/admissibility status and must not be
misused as an estimate of candidate merit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.config import CONFIG
from app.services.preprocessing import NormalizedTextBundle


@dataclass(slots=True)
class EligibilityResult:
    status: str
    reasons: list[str]


def _extract_material_counters(profile: dict[str, Any]) -> tuple[int, int, bool]:
    """Extract documents/portfolio/video counters from common profile locations."""
    docs_count = 0
    portfolio_links_count = 0
    has_video = False

    structured = profile.get("structured_data") if isinstance(profile.get("structured_data"), dict) else {}
    materials = structured.get("application_materials") if isinstance(structured.get("application_materials"), dict) else {}

    def _len_if_list(value: Any) -> int:
        return len(value) if isinstance(value, list) else 0

    docs_count += _len_if_list(profile.get("documents"))
    docs_count += _len_if_list(profile.get("attachments"))
    docs_count += _len_if_list(profile.get("supporting_documents"))
    docs_count += _len_if_list(materials.get("documents"))
    docs_count += _len_if_list(materials.get("attachments"))

    portfolio_links_count += _len_if_list(profile.get("portfolio_links"))
    portfolio_links_count += _len_if_list(materials.get("portfolio_links"))

    top_video = profile.get("video_presentation_link") or profile.get("videoPresentationLink") or profile.get("video_url")
    material_video = materials.get("video_presentation_link") or materials.get("videoPresentationLink") or materials.get("video_url")
    has_video = bool((isinstance(top_video, str) and top_video.strip()) or (isinstance(material_video, str) and material_video.strip()))

    return docs_count, portfolio_links_count, has_video


def evaluate_eligibility(
    candidate_id: str | None,
    consent: bool | None,
    bundle: NormalizedTextBundle,
    profile: dict[str, Any] | None = None,
) -> EligibilityResult:
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

    profile = profile or {}
    docs_count, portfolio_links_count, has_video = _extract_material_counters(profile)

    if CONFIG.thresholds.min_required_documents > 0 and docs_count < CONFIG.thresholds.min_required_documents:
        reasons.append("missing_required_materials_documents")

    if CONFIG.thresholds.min_portfolio_links > 0 and portfolio_links_count < CONFIG.thresholds.min_portfolio_links:
        reasons.append("missing_required_materials_portfolio")

    if CONFIG.thresholds.require_video_presentation and not has_video:
        reasons.append("missing_required_materials_video")

    logical_groups_present = int(bundle.stats.get("logical_source_groups_present", 0))
    logical_groups_total = int(bundle.stats.get("logical_source_groups_total", 3))
    logical_groups_missing = max(logical_groups_total - logical_groups_present, 0)

    if word_count < 20:
        return EligibilityResult(status="incomplete_application", reasons=reasons or ["critical_text_missing"])

    if logical_groups_present < 2:
        reasons.append("insufficient_multi_source_evidence")

    if logical_groups_missing >= 2 or reasons:
        return EligibilityResult(status="conditionally_eligible", reasons=reasons or ["high_missingness"])

    return EligibilityResult(status="eligible", reasons=[])
