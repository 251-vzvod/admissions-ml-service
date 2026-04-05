"""Structured claim-to-evidence extraction for committee-facing review."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from app.services.preprocessing import NormalizedTextBundle
from app.services.semantic_rubrics import SemanticEvidence, SemanticRubricResult
from app.utils.math_utils import clamp01, to_display_score, weighted_average_normalized


@dataclass(slots=True)
class ClaimEvidenceItem:
    claim: str
    support_level: str
    source: str
    snippet: str
    support_score: int
    rationale: str


@dataclass(slots=True)
class ClaimEvidenceResult:
    supported_claims: list[ClaimEvidenceItem]
    weakly_supported_claims: list[ClaimEvidenceItem]


DIMENSION_LABELS = {
    "leadership_potential": "Candidate shows early leadership and responsibility through concrete action.",
    "growth_trajectory": "Candidate demonstrates growth through challenge, adaptation, and reflection.",
    "motivation_authenticity": "Candidate's motivation is grounded in concrete reasons, values, and future direction.",
    "authenticity_groundedness": "Candidate's story is supported by specific details rather than only abstract statements.",
    "community_orientation": "Candidate connects learning and initiative with usefulness for other people or a real community problem.",
}


DIMENSION_RATIONALES = {
    "leadership_potential": "Supported by leadership-like action and coordination signals in the candidate text.",
    "growth_trajectory": "Supported by challenge-response and reflection signals across sections.",
    "motivation_authenticity": "Supported by program-fit and grounded motivation signals rather than polish alone.",
    "authenticity_groundedness": "Supported by concrete details, consistency, and non-generic evidence.",
    "community_orientation": "Supported by concrete contribution, responsibility, or community-problem framing in the candidate text.",
}


WEAK_DIMENSION_LABELS = {
    "leadership_potential": "Leadership signal exists, but its concrete impact is still under-supported.",
    "growth_trajectory": "Growth trajectory signal exists, but concrete outcomes are still thin.",
    "motivation_authenticity": "Motivation seems genuine, but the case for why this program matters is not yet fully grounded.",
    "authenticity_groundedness": "Some parts of the story are plausible, but the grounding is still weaker than ideal.",
    "community_orientation": "Community-oriented intent exists, but the concrete contribution to other people is still under-supported.",
}

SOURCE_RELIABILITY = {
    "interview_text": 0.88,
    "video_interview_transcript_text": 0.84,
    "motivation_questions": 0.76,
    "motivation_letter_text": 0.66,
    "video_presentation_transcript_text": 0.62,
}


def _snippet(text: str, max_len: int = 160) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len] + "..."


def _dimension_support_raw(
    *,
    dimension_score: float,
    review_signals: dict[str, float],
    bundle: NormalizedTextBundle,
) -> float:
    source_groups = clamp01(float(bundle.stats.get("logical_source_groups_present", 0)) / 3.0)
    return weighted_average_normalized(
        [
            (dimension_score, 0.38),
            (float(review_signals.get("evidence_signal", 0.0)), 0.28),
            (float(review_signals.get("authenticity_signal", 0.0)), 0.22),
            (source_groups, 0.06),
            (1.0 - float(review_signals.get("polish_risk_signal", 0.0)), 0.06),
        ]
    )


def _build_item(
    *,
    claim: str,
    evidence: SemanticEvidence,
    support_raw: float,
    support_level: str,
    rationale: str,
) -> ClaimEvidenceItem:
    return ClaimEvidenceItem(
        claim=claim,
        support_level=support_level,
        source=evidence.source,
        snippet=_snippet(evidence.snippet),
        support_score=to_display_score(support_raw),
        rationale=rationale,
    )


def _dedupe_claim_items(items: list[ClaimEvidenceItem]) -> list[ClaimEvidenceItem]:
    seen: set[tuple[str, str]] = set()
    deduped: list[ClaimEvidenceItem] = []
    for item in items:
        key = (" ".join(item.claim.lower().split()), item.source)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _best_llm_snippet(
    claim: str,
    source: str,
    direct_snippet: str,
    llm_evidence_spans: list[dict[str, str]] | None,
) -> str:
    snippet = _snippet(direct_snippet)
    if snippet:
        return snippet
    for span in llm_evidence_spans or []:
        if str(span.get("source", "")).strip() != source:
            continue
        candidate = _snippet(str(span.get("snippet", "") or span.get("text", "")))
        if candidate:
            return candidate
    return _snippet(claim)


def _llm_claim_support_raw(
    *,
    source: str,
    review_signals: dict[str, float],
    bundle: NormalizedTextBundle,
    has_direct_snippet: bool,
) -> float:
    source_groups = clamp01(float(bundle.stats.get("logical_source_groups_present", 0)) / 3.0)
    source_reliability = float(SOURCE_RELIABILITY.get(source, 0.58))
    return weighted_average_normalized(
        [
            (source_reliability, 0.34),
            (float(review_signals.get("evidence_signal", 0.0)), 0.28),
            (float(review_signals.get("authenticity_signal", 0.0)), 0.18),
            (source_groups, 0.10),
            (1.0 if has_direct_snippet else 0.45, 0.10),
        ],
        default=0.0,
    )


def _build_llm_claim_items(
    *,
    llm_claims: list[dict[str, str]] | None,
    support_level: str,
    rationale: str,
    review_signals: dict[str, float],
    bundle: NormalizedTextBundle,
    llm_evidence_spans: list[dict[str, str]] | None,
    min_support_raw: float,
) -> list[ClaimEvidenceItem]:
    items: list[ClaimEvidenceItem] = []
    for claim_item in llm_claims or []:
        claim = " ".join(str(claim_item.get("claim", "")).split()).strip()
        source = str(claim_item.get("source", "motivation_letter_text")).strip() or "motivation_letter_text"
        raw_snippet = str(claim_item.get("snippet", "")).strip()
        if not claim:
            continue
        snippet = _best_llm_snippet(claim, source, raw_snippet, llm_evidence_spans)
        support_raw = _llm_claim_support_raw(
            source=source,
            review_signals=review_signals,
            bundle=bundle,
            has_direct_snippet=bool(raw_snippet),
        )
        if support_raw < min_support_raw:
            continue
        items.append(
            ClaimEvidenceItem(
                claim=claim,
                support_level=support_level,
                source=source,
                snippet=snippet,
                support_score=to_display_score(support_raw),
                rationale=rationale,
            )
        )
    return items


def build_claim_evidence_map(
    *,
    bundle: NormalizedTextBundle,
    review_signals: dict[str, float],
    semantic_result: SemanticRubricResult,
    hidden_potential_score: int,
    evidence_coverage_score: int,
    trajectory_score: int,
    llm_strength_claims: list[dict[str, str]] | None = None,
    llm_gap_claims: list[dict[str, str]] | None = None,
    llm_uncertainty_claims: list[dict[str, str]] | None = None,
    llm_evidence_spans: list[dict[str, str]] | None = None,
) -> ClaimEvidenceResult:
    supported_claims: list[ClaimEvidenceItem] = []
    weakly_supported_claims: list[ClaimEvidenceItem] = []

    used_dimensions: set[str] = set()
    dimension_candidates: list[tuple[str, float, float]] = []
    for dimension in DIMENSION_LABELS:
        score = float(semantic_result.features.get(f"semantic_{dimension}", 0.0))
        support_raw = _dimension_support_raw(
            dimension_score=score,
            review_signals=review_signals,
            bundle=bundle,
        )
        dimension_candidates.append((dimension, score, support_raw))

    for dimension, score, support_raw in sorted(
        dimension_candidates,
        key=lambda item: (item[2], item[1]),
        reverse=True,
    ):
        evidence = semantic_result.evidence.get(dimension)
        if evidence is None:
            continue

        if score >= 0.48 and support_raw >= 0.52:
            supported_claims.append(
                _build_item(
                    claim=DIMENSION_LABELS[dimension],
                    evidence=evidence,
                    support_raw=support_raw,
                    support_level="strong" if support_raw >= 0.64 else "moderate",
                    rationale=DIMENSION_RATIONALES[dimension],
                )
            )
            used_dimensions.add(dimension)
        elif score >= 0.42 and support_raw >= 0.34:
            weakly_supported_claims.append(
                _build_item(
                    claim=WEAK_DIMENSION_LABELS[dimension],
                    evidence=evidence,
                    support_raw=support_raw,
                    support_level="weak",
                    rationale="Signal exists, but more concrete or cross-section evidence is needed.",
                )
            )

    hidden_signal_raw = weighted_average_normalized(
        [
            (hidden_potential_score / 100.0, 0.44),
            (trajectory_score / 100.0, 0.26),
            (evidence_coverage_score / 100.0, 0.20),
            (float(review_signals.get("authenticity_signal", 0.0)), 0.10),
        ]
    )
    if hidden_potential_score >= 35 and evidence_coverage_score >= 24:
        anchor_dimension = "growth_trajectory"
        if float(semantic_result.features.get("semantic_leadership_potential", 0.0)) > float(
            semantic_result.features.get("semantic_growth_trajectory", 0.0)
        ):
            anchor_dimension = "leadership_potential"
        anchor_evidence = semantic_result.evidence.get(anchor_dimension)
        if anchor_evidence is not None:
            hidden_item = _build_item(
                claim="Underlying growth and leadership signal appears stronger than the candidate's self-presentation.",
                evidence=anchor_evidence,
                support_raw=hidden_signal_raw,
                support_level="strong" if hidden_signal_raw >= 0.62 else "moderate",
                rationale="This is a shortlist-oriented inference built from trajectory, evidence coverage, and consistency rather than writing polish.",
            )
            supported_claims.insert(0, hidden_item)
    elif hidden_potential_score >= 28:
        anchor_evidence = semantic_result.evidence.get("growth_trajectory")
        if anchor_evidence is not None:
            weakly_supported_claims.insert(
                0,
                _build_item(
                    claim="There may be hidden potential here, but the supporting evidence is still too thin for a stronger claim.",
                    evidence=anchor_evidence,
                    support_raw=hidden_signal_raw,
                    support_level="weak",
                    rationale="The system sees trajectory upside, but the current evidence floor remains limited.",
                ),
            )

    llm_supported_claims = _build_llm_claim_items(
        llm_claims=llm_strength_claims,
        support_level="moderate",
        rationale="LLM extraction surfaced this strength from candidate text; support score reflects source grounding and current evidence reliability.",
        review_signals=review_signals,
        bundle=bundle,
        llm_evidence_spans=llm_evidence_spans,
        min_support_raw=0.40,
    )
    llm_weak_claims = _build_llm_claim_items(
        llm_claims=[*(llm_gap_claims or []), *(llm_uncertainty_claims or [])],
        support_level="weak",
        rationale="LLM extraction identified this as a review-relevant point that still needs stronger grounding or manual verification.",
        review_signals=review_signals,
        bundle=bundle,
        llm_evidence_spans=llm_evidence_spans,
        min_support_raw=0.26,
    )

    supported_claims = _dedupe_claim_items([*llm_supported_claims, *supported_claims])[:3]
    weakly_supported_claims = _dedupe_claim_items([*llm_weak_claims, *weakly_supported_claims])[:3]

    return ClaimEvidenceResult(
        supported_claims=supported_claims,
        weakly_supported_claims=weakly_supported_claims,
    )
