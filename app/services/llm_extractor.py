"""LLM-assisted text feature extractor orchestrator."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import CONFIG
from app.services.llm_client import LLMClientError, LLMRequest, build_llm_client
from app.services.llm_parser import LLMExtractionOutput, LLMParseError, parse_llm_extraction_json
from app.services.llm_prompts import SYSTEM_PROMPT, build_extraction_user_prompt
from app.services.preprocessing import NormalizedTextBundle


@dataclass(slots=True)
class LLMExtractionResult:
    features: dict[str, float | bool]
    diagnostics: dict[str, float | bool | int | str]
    strengths: list[str]
    gaps: list[str]
    uncertainties: list[str]
    evidence_spans: list[dict[str, str]]
    rationale: str
    llm_metadata: dict[str, str | int | float]


def _to_result(extraction: LLMExtractionOutput, llm_metadata: dict[str, str | int | float]) -> LLMExtractionResult:
    low_evidence_flag = extraction.evidence_count < 0.35
    return LLMExtractionResult(
        features={
            "motivation_clarity": extraction.motivation_clarity,
            "initiative": extraction.initiative,
            "leadership_impact": extraction.leadership_impact,
            "growth_trajectory": extraction.growth_trajectory,
            "resilience": extraction.resilience,
            "program_fit": extraction.program_fit,
            "evidence_richness": extraction.evidence_richness,
            "specificity_score": extraction.specificity_score,
            "evidence_count": extraction.evidence_count,
            "consistency_score": extraction.consistency_score,
            "completeness_score": extraction.completeness_score,
            "genericness_score": extraction.genericness_score,
            "contradiction_flag": extraction.contradiction_flag,
            "low_evidence_flag": low_evidence_flag,
            "polished_but_empty_score": extraction.polished_but_empty_score,
            "cross_section_mismatch_score": extraction.cross_section_mismatch_score,
        },
        diagnostics={
            "word_count": 0,
            "evidence_density": extraction.evidence_richness,
            "generic_density": extraction.genericness_score,
            "long_but_thin": extraction.polished_but_empty_score > 0.6,
            "extractor_rationale": extraction.extractor_rationale,
        },
        strengths=extraction.top_strength_signals[:3],
        gaps=extraction.main_gap_signals[:3],
        uncertainties=extraction.uncertainties[:3],
        evidence_spans=[span.model_dump() for span in extraction.evidence_spans[:3]],
        rationale=extraction.extractor_rationale,
        llm_metadata=llm_metadata,
    )


def extract_text_features_with_llm(bundle: NormalizedTextBundle) -> LLMExtractionResult:
    """Call configured LLM provider and parse structured extraction output."""
    llm_cfg = CONFIG.llm
    client = build_llm_client(provider=llm_cfg.provider, base_url=llm_cfg.base_url, api_key=llm_cfg.api_key)

    request = LLMRequest(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=build_extraction_user_prompt(bundle),
        model=llm_cfg.model,
        temperature=llm_cfg.temperature,
        timeout_seconds=llm_cfg.timeout_seconds,
        max_retries=llm_cfg.max_retries,
    )

    try:
        response = client.complete(request)
    except LLMClientError as exc:
        raise RuntimeError("llm_call_failed") from exc

    try:
        parsed = parse_llm_extraction_json(response.content)
    except LLMParseError as exc:
        raise RuntimeError("llm_parse_failed") from exc

    return _to_result(
        extraction=parsed,
        llm_metadata={
            "provider": response.provider,
            "model": response.model,
            "latency_ms": response.latency_ms,
        },
    )
