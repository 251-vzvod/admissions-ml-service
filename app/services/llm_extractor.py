"""LLM-assisted explainability extractor orchestrator."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import CONFIG
from app.services.llm_client import LLMClientError, LLMRequest, build_llm_client
from app.services.llm_parser import LLMExplainabilityOutput, LLMParseError, parse_llm_extraction_json
from app.services.llm_prompts import SYSTEM_PROMPT, build_extraction_user_prompt
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

    return _to_result(
        extraction=parsed,
        llm_metadata={
            "provider": response.provider,
            "model": response.model,
            "latency_ms": response.latency_ms,
        },
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
