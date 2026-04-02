"""LLM-assisted explainability extractor orchestrator."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import CONFIG
from app.services.llm_client import LLMClientError, LLMRequest, build_llm_client
from app.services.llm_parser import LLMExplainabilityOutput, LLMParseError, parse_llm_extraction_json
from app.services.llm_prompts import (
    PROMPT_VERSION,
    REPAIR_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_extraction_user_prompt,
    build_repair_user_prompt,
)
from app.services.preprocessing import NormalizedTextBundle


@dataclass(slots=True)
class LLMExplainabilityResult:
    strength_claims: list[dict[str, str]]
    gap_claims: list[dict[str, str]]
    uncertainty_claims: list[dict[str, str]]
    evidence_spans: list[dict[str, str]]
    rationale: str
    rubric_assessment: dict[str, int | str]
    committee_follow_up_question: str
    llm_metadata: dict[str, str | int | float]
    authenticity_assist: "LLMAuthenticityAssist"


@dataclass(slots=True)
class LLMAuthenticityAssist:
    available: bool
    review_needed: str
    risk_hint: float
    grounding_gap_score: float
    section_mismatch_score: float
    style_shift_score: float
    reasons: list[str]


_GROUNDING_HINT_TERMS = (
    "thin",
    "limited evidence",
    "insufficient evidence",
    "weak evidence",
    "not grounded",
    "under-supported",
    "unsupported",
    "unquantified",
    "generic",
    "vague",
    "not specific",
    "few details",
    "few examples",
    "little detail",
    "little evidence",
    "abstract",
)

_MISMATCH_HINT_TERMS = (
    "section",
    "sections",
    "interview",
    "essay",
    "q&a",
    "qa",
    "do not align",
    "does not align",
    "misalign",
    "mismatch",
    "inconsistent",
    "varies",
    "different emphasis",
    "different role",
    "different timeline",
)

_STYLE_SHIFT_HINT_TERMS = (
    "tone",
    "voice",
    "register",
    "style shift",
    "more polished",
    "over-polished",
    "formulaic",
    "scripted",
    "rewritten",
    "consultant",
)


def _has_substantive_extraction(extraction: LLMExplainabilityOutput) -> bool:
    return any(
        [
            bool(extraction.top_strength_signals),
            bool(extraction.main_gap_signals),
            bool(extraction.uncertainty_signals),
            bool(extraction.evidence_spans),
            bool(extraction.rubric_assessment),
            bool(extraction.committee_follow_up_question),
        ]
    )


def _bounded_score_from_unit_interval(value: float) -> int:
    if value < 0.2:
        return 1
    if value < 0.4:
        return 2
    if value < 0.6:
        return 3
    if value < 0.8:
        return 4
    return 5


def _fallback_rubric_from_signals(deterministic_signals: dict[str, float | bool] | None) -> dict[str, int | str]:
    features = deterministic_signals or {}
    initiative = float(features.get("initiative", 0.0))
    leadership = float(features.get("leadership_impact", 0.0))
    growth = float(features.get("growth_trajectory", 0.0))
    resilience = float(features.get("resilience", 0.0))
    motivation = float(features.get("motivation_clarity", 0.0))
    program_fit = float(features.get("program_fit", 0.0))
    evidence = float(features.get("evidence_richness", 0.0))
    specificity = float(features.get("specificity_score", 0.0))
    evidence_count = float(features.get("evidence_count", 0.0))
    community = float(features.get("community_value_orientation", 0.0))
    polish_risk = float(features.get("polished_but_empty_score", 0.0))
    mismatch = float(features.get("cross_section_mismatch_score", 0.0))
    consistency = float(features.get("consistency_score", 0.0))
    contradiction = bool(features.get("contradiction_flag", False))

    leadership_signal = (initiative * 0.45) + (leadership * 0.35) + (community * 0.10) + (growth * 0.10)
    motivation_signal = (motivation * 0.45) + (program_fit * 0.25) + (evidence * 0.20) + ((1.0 - polish_risk) * 0.10)
    evidence_signal = (evidence * 0.50) + (specificity * 0.30) + (evidence_count * 0.20)
    hidden_signal = (growth * 0.35) + (initiative * 0.25) + (resilience * 0.20) + ((1.0 - polish_risk) * 0.20)

    authenticity_risk = max(polish_risk, mismatch, 1.0 - consistency)
    if contradiction or authenticity_risk >= 0.65:
        authenticity_band = "high"
    elif authenticity_risk >= 0.40:
        authenticity_band = "medium"
    else:
        authenticity_band = "low"

    return {
        "leadership_potential": _bounded_score_from_unit_interval(leadership_signal),
        "growth_trajectory": _bounded_score_from_unit_interval(growth),
        "motivation_authenticity": _bounded_score_from_unit_interval(motivation_signal),
        "evidence_strength": _bounded_score_from_unit_interval(evidence_signal),
        "hidden_potential_hint": _bounded_score_from_unit_interval(hidden_signal),
        "authenticity_review_needed": authenticity_band,
    }


def _claim_to_dict(item: dict[str, str] | object) -> dict[str, str]:
    if isinstance(item, dict):
        return {
            "claim": str(item.get("claim", "")).strip(),
            "source": str(item.get("source", "motivation_letter_text")).strip() or "motivation_letter_text",
            "snippet": str(item.get("snippet", "")).strip(),
        }
    return {"claim": "", "source": "motivation_letter_text", "snippet": ""}


def _score_from_five_point_reverse(raw: int | str | None) -> float:
    try:
        value = int(raw) if raw is not None else 3
    except (TypeError, ValueError):
        value = 3
    bounded = max(1, min(5, value))
    return (5 - bounded) / 4.0


def _band_to_unit_risk(raw: str | None) -> float:
    lowered = str(raw or "").strip().lower()
    if lowered == "high":
        return 0.82
    if lowered == "medium":
        return 0.48
    return 0.14


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _derive_authenticity_assist(
    extraction: LLMExplainabilityOutput,
    llm_metadata: dict[str, str | int | float],
) -> LLMAuthenticityAssist:
    rubric = extraction.rubric_assessment or {}
    review_needed = str(rubric.get("authenticity_review_needed", "low")).strip().lower() or "low"
    evidence_gap = _score_from_five_point_reverse(rubric.get("evidence_strength"))
    motivation_gap = _score_from_five_point_reverse(rubric.get("motivation_authenticity"))

    claims = [
        *(item.claim for item in extraction.main_gap_signals),
        *(item.claim for item in extraction.uncertainty_signals),
    ]
    rationale = extraction.extractor_rationale or ""

    grounding_reasons: list[str] = []
    mismatch_reasons: list[str] = []
    style_reasons: list[str] = []
    for claim in claims:
        normalized = claim.strip()
        if not normalized:
            continue
        if _contains_any(normalized, _GROUNDING_HINT_TERMS):
            grounding_reasons.append(normalized)
        if _contains_any(normalized, _MISMATCH_HINT_TERMS):
            mismatch_reasons.append(normalized)
        if _contains_any(normalized, _STYLE_SHIFT_HINT_TERMS):
            style_reasons.append(normalized)

    if _contains_any(rationale, _MISMATCH_HINT_TERMS):
        mismatch_reasons.append(rationale.strip())
    if _contains_any(rationale, _STYLE_SHIFT_HINT_TERMS):
        style_reasons.append(rationale.strip())
    if _contains_any(rationale, _GROUNDING_HINT_TERMS):
        grounding_reasons.append(rationale.strip())

    band_risk = _band_to_unit_risk(review_needed)
    grounding_gap_score = min(1.0, (evidence_gap * 0.75) + (motivation_gap * 0.25) + (0.18 if grounding_reasons else 0.0))
    section_mismatch_score = min(1.0, band_risk * 0.55 + (0.35 if mismatch_reasons else 0.0))
    style_shift_score = min(1.0, band_risk * 0.30 + (0.55 if style_reasons else 0.0))
    risk_hint = min(
        1.0,
        (band_risk * 0.42)
        + (grounding_gap_score * 0.34)
        + (section_mismatch_score * 0.16)
        + (style_shift_score * 0.08),
    )

    substantive = any(
        [
            bool(extraction.main_gap_signals),
            bool(extraction.uncertainty_signals),
            bool(extraction.rubric_assessment),
            bool(extraction.extractor_rationale),
        ]
    )
    if llm_metadata.get("deterministic_rubric_fallback") == "true" and not extraction.main_gap_signals and not extraction.uncertainty_signals:
        substantive = False

    reasons = [*grounding_reasons[:2], *mismatch_reasons[:1], *style_reasons[:1]]
    if not reasons and review_needed in {"medium", "high"}:
        reasons.append(f"LLM review signal suggested {review_needed} authenticity review need.")

    return LLMAuthenticityAssist(
        available=substantive,
        review_needed=review_needed,
        risk_hint=risk_hint,
        grounding_gap_score=grounding_gap_score,
        section_mismatch_score=section_mismatch_score,
        style_shift_score=style_shift_score,
        reasons=reasons[:3],
    )


def _to_result(extraction: LLMExplainabilityOutput, llm_metadata: dict[str, str | int | float]) -> LLMExplainabilityResult:
    strength_claims = [_claim_to_dict(item.model_dump()) for item in extraction.top_strength_signals[:3]]
    gap_claims = [_claim_to_dict(item.model_dump()) for item in extraction.main_gap_signals[:3]]
    uncertainty_claims = [_claim_to_dict(item.model_dump()) for item in extraction.uncertainty_signals[:3]]

    if not strength_claims and extraction.evidence_bullets:
        strength_claims = [
            {
                "claim": bullet,
                "source": "motivation_letter_text",
                "snippet": "",
            }
            for bullet in extraction.evidence_bullets[:3]
        ]

    if not uncertainty_claims and extraction.uncertainties:
        uncertainty_claims = [
            {
                "claim": item,
                "source": "motivation_letter_text",
                "snippet": "",
            }
            for item in extraction.uncertainties[:3]
        ]

    evidence_spans = [
        {
            "dimension": span.dimension,
            "source": span.source,
            "snippet": span.text,
        }
        for span in extraction.evidence_spans[:4]
    ]

    if not evidence_spans:
        fallback_spans: list[dict[str, str]] = []
        for item in [*strength_claims, *gap_claims, *uncertainty_claims]:
            snippet = item.get("snippet", "")
            if not snippet:
                continue
            fallback_spans.append(
                {
                    "dimension": "explainability",
                    "source": item.get("source", "motivation_letter_text"),
                    "snippet": snippet,
                }
            )
            if len(fallback_spans) >= 4:
                break
        evidence_spans = fallback_spans

    return LLMExplainabilityResult(
        strength_claims=strength_claims,
        gap_claims=gap_claims,
        uncertainty_claims=uncertainty_claims,
        evidence_spans=evidence_spans,
        rationale=extraction.extractor_rationale or (extraction.human_review.notes if extraction.human_review else ""),
        rubric_assessment=dict(extraction.rubric_assessment),
        committee_follow_up_question=extraction.committee_follow_up_question,
        llm_metadata=llm_metadata,
        authenticity_assist=_derive_authenticity_assist(extraction, llm_metadata),
    )


def extract_explainability_with_llm(
    bundle: NormalizedTextBundle,
    deterministic_signals: dict[str, float | bool] | None = None,
) -> LLMExplainabilityResult:
    """Call configured LLM provider and parse explainability-only output."""
    llm_cfg = CONFIG.llm
    client = build_llm_client(provider=llm_cfg.provider, base_url=llm_cfg.base_url, api_key=llm_cfg.api_key)

    request = LLMRequest(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_extraction_user_prompt(
            bundle=bundle,
            deterministic_signals=deterministic_signals,
        ),
        model=llm_cfg.model,
        temperature=llm_cfg.temperature,
        timeout_seconds=llm_cfg.timeout_seconds,
        max_retries=llm_cfg.max_retries,
        retry_backoff_seconds=llm_cfg.retry_backoff_seconds,
        retry_jitter_seconds=llm_cfg.retry_jitter_seconds,
    )

    try:
        response = client.complete(request)
    except LLMClientError as exc:
        raise RuntimeError("llm_call_failed") from exc

    try:
        parsed = parse_llm_extraction_json(response.content)
    except LLMParseError as exc:
        raise RuntimeError(f"llm_parse_failed:{exc}") from exc

    llm_metadata: dict[str, str | int | float] = {
        "provider": response.provider,
        "model": response.model,
        "latency_ms": response.latency_ms,
        "prompt_version": CONFIG.prompt_version or PROMPT_VERSION,
    }

    if not _has_substantive_extraction(parsed):
        repair_request = LLMRequest(
            system_prompt=REPAIR_SYSTEM_PROMPT,
            user_prompt=build_repair_user_prompt(
                bundle=bundle,
                prior_response=response.content,
                deterministic_signals=deterministic_signals,
            ),
            model=llm_cfg.model,
            temperature=0.0,
            timeout_seconds=llm_cfg.timeout_seconds,
            max_retries=llm_cfg.max_retries,
            retry_backoff_seconds=llm_cfg.retry_backoff_seconds,
            retry_jitter_seconds=llm_cfg.retry_jitter_seconds,
        )
        llm_metadata["repair_attempted"] = "true"
        try:
            repair_response = client.complete(repair_request)
            repaired = parse_llm_extraction_json(repair_response.content)
            llm_metadata["repair_latency_ms"] = repair_response.latency_ms
            if _has_substantive_extraction(repaired):
                parsed = repaired
                llm_metadata["repair_used"] = "true"
            else:
                llm_metadata["repair_used"] = "false"
        except (LLMClientError, LLMParseError):
            llm_metadata["repair_used"] = "false"

    if not parsed.rubric_assessment:
        parsed.rubric_assessment = _fallback_rubric_from_signals(deterministic_signals)
        llm_metadata["deterministic_rubric_fallback"] = "true"

    return _to_result(
        extraction=parsed,
        llm_metadata=llm_metadata,
    )


def extract_text_features_with_llm(
    bundle: NormalizedTextBundle,
    deterministic_signals: dict[str, float | bool] | None = None,
) -> LLMExplainabilityResult:
    """Backward-compatible alias to explainability-only extraction."""
    return extract_explainability_with_llm(
        bundle=bundle,
        deterministic_signals=deterministic_signals,
    )
