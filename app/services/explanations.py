"""Explanation layer for transparent committee-facing narratives."""

from __future__ import annotations

from dataclasses import dataclass

from app.schemas.decision import Recommendation, ReviewFlag
from app.schemas.output import EvidenceSpan, ExplanationNotes, ExplanationPayload


@dataclass(slots=True)
class ExplanationResult:
    top_strengths: list[str]
    main_gaps: list[str]
    uncertainties: list[str]
    evidence_spans: list[EvidenceSpan]
    explanation: ExplanationPayload


def _pick_snippet(text: str, max_len: int = 180) -> str:
    cleaned = " ".join(text.split())
    return cleaned[:max_len] + ("..." if len(cleaned) > max_len else "")


def _format_claim_with_evidence(item: dict[str, str]) -> str:
    claim = " ".join((item.get("claim") or "").split()).strip()
    source = (item.get("source") or "motivation_letter_text").strip() or "motivation_letter_text"
    snippet = _pick_snippet(item.get("snippet") or "", max_len=120)
    source_label = {
        "motivation_letter_text": "essay",
        "motivation_questions": "Q&A",
        "interview_text": "interview",
        "video_interview_transcript_text": "video interview",
        "video_presentation_transcript_text": "video presentation",
    }.get(source, "application")

    if claim and snippet:
        return f'{claim} Evidence from {source_label}: "{snippet}"'
    if claim:
        return claim
    if snippet:
        return f'Evidence from {source_label}: "{snippet}"'
    return ""


def _normalize_uncertainty_claim(item: dict[str, str]) -> dict[str, str]:
    claim = " ".join((item.get("claim") or "").split()).strip()
    lowered = claim.lower()
    predictive_markers = [
        "future",
        "potential",
        "capability",
        "will",
        "may become",
    ]
    if claim and any(marker in lowered for marker in predictive_markers) and "evidence" not in lowered:
        claim = f"insufficient evidence to confirm: {claim}"
    normalized = dict(item)
    normalized["claim"] = claim
    return normalized


def _recommendation_summary_phrase(recommendation: Recommendation | str) -> str:
    if recommendation == Recommendation.REVIEW_PRIORITY:
        return "This looks like a strong candidate who should be reviewed with priority."
    if recommendation == Recommendation.STANDARD_REVIEW:
        return "This candidate shows some promising signals, but the case is not strong enough to fast-track."
    if recommendation == Recommendation.MANUAL_REVIEW_REQUIRED:
        return "This candidate may be promising, but the profile needs closer manual review before the committee can trust it."
    if recommendation == Recommendation.INSUFFICIENT_EVIDENCE:
        return "There is not enough concrete evidence yet for a confident committee decision."
    if recommendation == Recommendation.INCOMPLETE_APPLICATION:
        return "The application is incomplete, so the committee cannot assess it fairly yet."
    if recommendation == Recommendation.INVALID:
        return "The application does not currently meet the minimum requirements for substantive review."
    return "This profile needs committee review."


def _score_band(value: int | None) -> str:
    if value is None:
        return "unknown"
    if value >= 70:
        return "strong"
    if value >= 55:
        return "solid"
    if value >= 40:
        return "mixed"
    return "weak"


def _build_plain_language_summary(
    recommendation: Recommendation | str,
    merit_score: int | None,
    confidence_score: int | None,
    authenticity_risk: int | None,
    review_signals: dict[str, float],
) -> str:
    lead = _recommendation_summary_phrase(recommendation)

    growth = float(review_signals.get("growth_signal", 0.0))
    agency = float(review_signals.get("agency_signal", 0.0))
    evidence = float(review_signals.get("evidence_signal", 0.0))
    authenticity = float(review_signals.get("authenticity_signal", 0.0))
    hidden = float(review_signals.get("hidden_signal", 0.0))

    strengths: list[str] = []
    if growth >= 0.62:
        strengths.append("clear growth and reflection")
    if agency >= 0.56:
        strengths.append("real initiative or ownership")
    if evidence >= 0.60:
        strengths.append("enough concrete evidence to support the case")
    if hidden >= 0.58:
        strengths.append("stronger underlying potential than presentation quality")

    concerns: list[str] = []
    if evidence < 0.45:
        concerns.append("the evidence is still quite thin")
    if authenticity_risk is not None and authenticity_risk >= 45:
        concerns.append("some parts of the application need closer manual verification")
    elif authenticity < 0.52:
        concerns.append("there are some credibility or consistency questions to check")
    if confidence_score is not None and confidence_score < 50:
        concerns.append("the current assessment is only moderately reliable")

    middle_parts: list[str] = []
    if strengths:
        joined_strengths = ", ".join(strengths[:2]) if len(strengths) > 1 else strengths[0]
        middle_parts.append(f"The main positive signals are {joined_strengths}.")
    else:
        merit_band = _score_band(merit_score)
        middle_parts.append(f"Overall substance currently looks {merit_band}.")

    if concerns:
        joined_concerns = ", ".join(concerns[:2]) if len(concerns) > 1 else concerns[0]
        middle_parts.append(f"The main caution is that {joined_concerns}.")

    metrics_sentence = ""
    if merit_score is not None and confidence_score is not None and authenticity_risk is not None:
        metrics_sentence = (
            f" Current scores: merit {merit_score}/100, confidence {confidence_score}/100, "
            f"authenticity risk {authenticity_risk}/100."
        )

    return " ".join([lead, *middle_parts]).strip() + metrics_sentence


def build_explanation(
    review_signals: dict[str, float],
    merit_breakdown: dict[str, int],
    recommendation: Recommendation | str,
    review_flags: list[ReviewFlag | str],
    sections: dict[str, list[str]],
    extraction_mode: str = "hybrid",
    merit_score: int | None = None,
    confidence_score: int | None = None,
    authenticity_risk: int | None = None,
    provided_strengths: list[str] | None = None,
    provided_gaps: list[str] | None = None,
    provided_uncertainties: list[str] | None = None,
    provided_strength_claims: list[dict[str, str]] | None = None,
    provided_gap_claims: list[dict[str, str]] | None = None,
    provided_uncertainty_claims: list[dict[str, str]] | None = None,
    provided_evidence_spans: list[dict[str, str]] | None = None,
    extractor_rationale: str | None = None,
) -> ExplanationResult:
    """Build strengths/gaps/uncertainties and natural language scoring notes."""
    strengths: list[str] = []
    gaps: list[str] = []
    uncertainties: list[str] = []

    growth = float(review_signals.get("growth_signal", 0.0))
    agency = float(review_signals.get("agency_signal", 0.0))
    motivation = float(review_signals.get("motivation_signal", 0.0))
    community = float(review_signals.get("community_signal", 0.0))
    evidence = float(review_signals.get("evidence_signal", 0.0))
    authenticity = float(review_signals.get("authenticity_signal", 0.0))
    polish_risk = float(review_signals.get("polish_risk_signal", 0.0))
    hidden = float(review_signals.get("hidden_signal", 0.0))

    if growth >= 0.62:
        strengths.append("Strong growth trajectory signals supported by temporal and reflective evidence.")
    if agency >= 0.56:
        strengths.append("Clear initiative and agency markers in self-driven actions.")
    if motivation >= 0.58:
        strengths.append("Motivation aligns with the program format and collaborative learning context.")
    if community >= 0.52:
        strengths.append("Profile shows community-oriented motivation grounded in responsibility toward other people.")
    if evidence >= 0.60:
        strengths.append("Evidence density is sufficient to support a comparatively reliable assessment.")
    if hidden >= 0.58:
        strengths.append("Profile shows hidden-potential characteristics: growth and agency signals stronger than presentation quality.")

    if evidence < 0.45:
        gaps.append("Specificity is limited; more concrete examples and outcomes would improve reliability.")
    if evidence < 0.38:
        gaps.append("Evidence density is low relative to answer length.")
    if authenticity < 0.44:
        gaps.append("Potential internal inconsistency detected across sections.")
    if evidence < 0.40:
        gaps.append("Application completeness is moderate/low, which limits confidence in the assessment.")
    if motivation < 0.42:
        gaps.append("Motivation appears under-grounded; committee should probe why this program specifically matters.")
    if community < 0.35:
        gaps.append("Contribution to other people or a real community problem is still under-evidenced.")

    if polish_risk > 0.55:
        uncertainties.append("Some text appears generic; committee should verify authenticity of claims.")
    if ReviewFlag.SECTION_MISMATCH in review_flags:
        uncertainties.append("Section mismatch risk: claims may not be consistently supported across sources.")
    if recommendation in {Recommendation.MANUAL_REVIEW_REQUIRED, Recommendation.INSUFFICIENT_EVIDENCE}:
        uncertainties.append("Human review is recommended before high-confidence prioritization.")
    if evidence < 0.50:
        uncertainties.append("Some claims are directionally promising but under-supported by concrete examples.")
    if authenticity < 0.52:
        uncertainties.append("Authenticity risk signals are elevated enough to warrant closer manual reading.")

    evidence_spans: list[EvidenceSpan] = []
    if provided_evidence_spans:
        for span in provided_evidence_spans[:2]:
            source = span.get("source", "motivation_letter_text")
            text = span.get("text") or span.get("snippet") or ""
            evidence_spans.append(EvidenceSpan(source=source, snippet=_pick_snippet(text)))
    else:
        for source in [
            "motivation_letter_text",
            "motivation_questions",
            "interview_text",
            "video_interview_transcript_text",
            "video_presentation_transcript_text",
        ]:
            source_sections = sections.get(source, [])
            if source_sections:
                evidence_spans.append(EvidenceSpan(source=source, snippet=_pick_snippet(source_sections[0])))
            if len(evidence_spans) >= 2:
                break

    summary = _build_plain_language_summary(
        recommendation=recommendation,
        merit_score=merit_score,
        confidence_score=confidence_score,
        authenticity_risk=authenticity_risk,
        review_signals=review_signals,
    )

    scoring_notes = ExplanationNotes(
        potential=(
            "Potential axis reflects growth, resilience, initiative, and fit signals; "
            f"current potential breakdown score is {merit_breakdown.get('potential', 0)}/100."
        ),
        motivation=(
            "Motivation axis combines clarity, fit, and grounded evidence; "
            f"motivation breakdown score is {merit_breakdown.get('motivation', 0)}/100."
        ),
        confidence=(
            "Confidence score measures reliability of this assessment, not candidate quality. "
            "It is affected by specificity, evidence count, consistency, and completeness."
        ),
        authenticity_risk=(
            "Authenticity risk is a review-risk signal (not cheating proof). "
            "Elevated risk triggers flags and manual routing, not automatic rejection."
            + (f" Extractor note: {extractor_rationale}" if extractor_rationale else "")
        ),
        recommendation=(
            "Recommendation is a workflow routing label for committee review and must not be interpreted as final admission decision."
        ),
    )

    llm_strengths = [_format_claim_with_evidence(item) for item in (provided_strength_claims or [])]
    llm_gaps = [_format_claim_with_evidence(item) for item in (provided_gap_claims or [])]
    llm_uncertainties = [
        _format_claim_with_evidence(_normalize_uncertainty_claim(item)) for item in (provided_uncertainty_claims or [])
    ]

    top_strengths_final = [item for item in llm_strengths if item][:3] if llm_strengths else (provided_strengths[:3] if provided_strengths else strengths[:3])
    main_gaps_final = [item for item in llm_gaps if item][:3] if llm_gaps else (provided_gaps[:3] if provided_gaps else gaps[:3])
    uncertainties_final = (
        [item for item in llm_uncertainties if item][:3]
        if llm_uncertainties
        else (provided_uncertainties[:3] if provided_uncertainties else uncertainties[:3])
    )

    return ExplanationResult(
        top_strengths=top_strengths_final,
        main_gaps=main_gaps_final,
        uncertainties=uncertainties_final,
        evidence_spans=evidence_spans,
        explanation=ExplanationPayload(summary=summary, scoring_notes=scoring_notes),
    )
