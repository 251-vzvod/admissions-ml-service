"""Second-pass LLM writer for committee-facing public narrative fields."""

from __future__ import annotations

from dataclasses import dataclass

from app.config import CONFIG
from app.services.llm_client import LLMClientError, LLMRequest, build_llm_client
from app.services.llm_parser import (
    LLMCommitteeNarrativeOutput,
    LLMParseError,
    parse_llm_committee_json,
)
from app.services.llm_prompts import (
    COMMITTEE_WRITER_SYSTEM_PROMPT,
    build_committee_writer_user_prompt,
)
from app.services.preprocessing import NormalizedTextBundle


@dataclass(slots=True)
class LLMCommitteeNarrativeResult:
    summary: str
    top_strengths: list[str]
    main_gaps: list[str]
    what_to_verify_manually: list[str]
    suggested_follow_up_question: str
    llm_metadata: dict[str, str | int | float]


def _to_result(
    output: LLMCommitteeNarrativeOutput,
    llm_metadata: dict[str, str | int | float],
) -> LLMCommitteeNarrativeResult:
    return LLMCommitteeNarrativeResult(
        summary=output.summary,
        top_strengths=output.top_strengths[:3],
        main_gaps=output.main_gaps[:3],
        what_to_verify_manually=output.what_to_verify_manually[:3],
        suggested_follow_up_question=output.suggested_follow_up_question,
        llm_metadata=llm_metadata,
    )


def generate_committee_narrative_with_llm(
    *,
    detail_level: str,
    candidate_id: str,
    recommendation: str,
    merit_score: int,
    confidence_score: int,
    authenticity_risk: int,
    review_flags: list[str],
    committee_cohorts: list[str],
    why_candidate_surfaced: list[str],
    what_to_verify_manually: list[str],
    suggested_follow_up_question: str,
    top_strengths: list[str],
    main_gaps: list[str],
    uncertainties: list[str],
    evidence_highlights: list[dict[str, str]],
    bundle: NormalizedTextBundle,
) -> LLMCommitteeNarrativeResult:
    if not CONFIG.llm.enabled:
        raise RuntimeError("llm_disabled")

    client = build_llm_client(
        provider=CONFIG.llm.provider,
        base_url=CONFIG.llm.base_url,
        api_key=CONFIG.llm.api_key,
    )
    user_prompt = build_committee_writer_user_prompt(
        detail_level=detail_level,
        candidate_id=candidate_id,
        recommendation=recommendation,
        merit_score=merit_score,
        confidence_score=confidence_score,
        authenticity_risk=authenticity_risk,
        review_flags=review_flags,
        committee_cohorts=committee_cohorts,
        why_candidate_surfaced=why_candidate_surfaced,
        what_to_verify_manually=what_to_verify_manually,
        suggested_follow_up_question=suggested_follow_up_question,
        top_strengths=top_strengths,
        main_gaps=main_gaps,
        uncertainties=uncertainties,
        evidence_highlights=evidence_highlights,
        bundle=bundle,
    )

    request = LLMRequest(
        system_prompt=COMMITTEE_WRITER_SYSTEM_PROMPT,
        user_prompt=user_prompt,
        model=CONFIG.llm.model,
        temperature=CONFIG.llm.temperature,
        timeout_seconds=CONFIG.llm.timeout_seconds,
        max_retries=CONFIG.llm.max_retries,
        retry_backoff_seconds=CONFIG.llm.retry_backoff_seconds,
        retry_jitter_seconds=CONFIG.llm.retry_jitter_seconds,
    )

    try:
        response = client.complete(request)
        parsed = parse_llm_committee_json(response.content)
    except (LLMClientError, LLMParseError) as exc:
        raise RuntimeError("committee_writer_failed") from exc

    return _to_result(
        parsed,
        {
            "provider": response.provider,
            "model": response.model,
            "latency_ms": response.latency_ms,
            "writer_mode": "committee_narrative_v1",
        },
    )
