"""LLM-assisted explainability extractor orchestrator."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import CONFIG
from app.services.llm_client import LLMClientError, LLMRequest, build_llm_client
from app.services.llm_parser import LLMExplainabilityOutput, LLMParseError, parse_llm_extraction_json
from app.services.llm_prompts import (
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
