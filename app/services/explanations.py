"""Explanation layer for transparent committee-facing narratives."""

from __future__ import annotations

from dataclasses import dataclass

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


def build_explanation(
    feature_map: dict[str, float | bool],
    merit_breakdown: dict[str, int],
    recommendation: str,
    review_flags: list[str],
    sections: dict[str, list[str]],
    provided_strengths: list[str] | None = None,
    provided_gaps: list[str] | None = None,
    provided_uncertainties: list[str] | None = None,
    provided_evidence_spans: list[dict[str, str]] | None = None,
    extractor_rationale: str | None = None,
) -> ExplanationResult:
    """Build strengths/gaps/uncertainties and natural language scoring notes."""
    strengths: list[str] = []
    gaps: list[str] = []
    uncertainties: list[str] = []

    if float(feature_map.get("growth_trajectory", 0.0)) >= 0.65:
        strengths.append("Strong growth trajectory signals supported by temporal and reflective evidence.")
    if float(feature_map.get("initiative", 0.0)) >= 0.60:
        strengths.append("Clear initiative and agency markers in self-driven actions.")
    if float(feature_map.get("program_fit", 0.0)) >= 0.60:
        strengths.append("Motivation aligns with the program format and collaborative learning context.")

    if float(feature_map.get("specificity_score", 0.0)) < 0.45:
        gaps.append("Specificity is limited; more concrete examples and outcomes would improve reliability.")
    if float(feature_map.get("evidence_count", 0.0)) < 0.40:
        gaps.append("Evidence density is low relative to answer length.")
    if bool(feature_map.get("contradiction_flag", False)):
        gaps.append("Potential internal inconsistency detected across sections.")

    if float(feature_map.get("genericness_score", 0.0)) > 0.55:
        uncertainties.append("Some text appears generic; committee should verify authenticity of claims.")
    if "section_mismatch" in review_flags:
        uncertainties.append("Section mismatch risk: claims may not be consistently supported across sources.")
    if recommendation in {"manual_review_required", "insufficient_evidence"}:
        uncertainties.append("Human review is recommended before high-confidence prioritization.")

    evidence_spans: list[EvidenceSpan] = []
    if provided_evidence_spans:
        for span in provided_evidence_spans[:2]:
            source = span.get("source", "motivation_letter_text")
            text = span.get("text") or span.get("snippet") or ""
            evidence_spans.append(EvidenceSpan(source=source, snippet=_pick_snippet(text)))
    else:
        for source in ["motivation_letter_text", "motivation_questions", "interview_text"]:
            source_sections = sections.get(source, [])
            if source_sections:
                evidence_spans.append(EvidenceSpan(source=source, snippet=_pick_snippet(source_sections[0])))
            if len(evidence_spans) >= 2:
                break

    summary = (
        "This profile was scored with a deterministic heuristic baseline that separates candidate potential signals "
        "(merit), assessment reliability (confidence), and review uncertainty (authenticity risk). "
        f"Current routing recommendation: {recommendation}."
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

    return ExplanationResult(
        top_strengths=(provided_strengths[:3] if provided_strengths else strengths[:3]),
        main_gaps=(provided_gaps[:3] if provided_gaps else gaps[:3]),
        uncertainties=(provided_uncertainties[:3] if provided_uncertainties else uncertainties[:3]),
        evidence_spans=evidence_spans,
        explanation=ExplanationPayload(summary=summary, scoring_notes=scoring_notes),
    )
