"""Calibration report helpers built on top of the current scoring pipeline."""

from __future__ import annotations

from typing import Any

from app.services.pipeline import ScoringPipeline
from app.services.policy import build_policy_snapshot


def _human_review_value(review: dict[str, Any], key: str) -> bool:
    return bool(review.get(key, False))


def compare_cases(cases: list[object]) -> dict[str, Any]:
    pipeline = ScoringPipeline()
    comparisons: list[dict[str, Any]] = []

    recommendation_matches = 0
    shortlist_matches = 0
    hidden_matches = 0
    support_matches = 0
    authenticity_matches = 0

    for case in cases:
        candidate_id = str(getattr(case, "candidate_id", ""))
        candidate_payload = getattr(case, "candidate_payload", {})
        human_review = getattr(case, "human_review", {}) or {}

        result = pipeline.score_candidate(candidate_payload)
        model_policy = build_policy_snapshot(
            merit_score=result.merit_score,
            confidence_score=result.confidence_score,
            authenticity_risk=result.authenticity_risk,
            hidden_potential_score=result.hidden_potential_score,
            support_needed_score=result.support_needed_score,
            shortlist_priority_score=result.shortlist_priority_score,
            evidence_coverage_score=result.evidence_coverage_score,
            trajectory_score=result.trajectory_score,
        )

        comparison = {
            "candidate_id": candidate_id,
            "model": {
                "recommendation": result.recommendation,
                "shortlist_band": model_policy.shortlist_band,
                "hidden_potential_band": model_policy.hidden_potential_band,
                "support_needed_band": model_policy.support_needed_band,
                "authenticity_review_band": model_policy.authenticity_review_band,
                "merit_score": result.merit_score,
                "confidence_score": result.confidence_score,
                "authenticity_risk": result.authenticity_risk,
            },
            "human_review": human_review,
            "notes": human_review.get("notes", ""),
        }
        comparisons.append(comparison)

        recommendation_matches += int(result.recommendation == human_review.get("recommendation"))
        shortlist_matches += int(model_policy.shortlist_band == _human_review_value(human_review, "shortlist_band"))
        hidden_matches += int(
            model_policy.hidden_potential_band == _human_review_value(human_review, "hidden_potential_band")
        )
        support_matches += int(
            model_policy.support_needed_band == _human_review_value(human_review, "support_needed_band")
        )
        authenticity_matches += int(
            model_policy.authenticity_review_band == _human_review_value(human_review, "authenticity_review_band")
        )

    case_count = len(comparisons)
    denominator = case_count or 1
    return {
        "summary": {
            "case_count": case_count,
            "recommendation_match_rate": round(recommendation_matches / denominator, 6),
            "shortlist_band_match_rate": round(shortlist_matches / denominator, 6),
            "hidden_potential_match_rate": round(hidden_matches / denominator, 6),
            "support_needed_match_rate": round(support_matches / denominator, 6),
            "authenticity_review_match_rate": round(authenticity_matches / denominator, 6),
        },
        "comparisons": comparisons,
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary", {}) if isinstance(report, dict) else {}
    comparisons = report.get("comparisons", []) if isinstance(report, dict) else []

    lines = [
        "# Calibration Report",
        "",
        f"- Cases: {summary.get('case_count', 0)}",
        f"- Recommendation match rate: {summary.get('recommendation_match_rate', 0.0)}",
        f"- Shortlist band match rate: {summary.get('shortlist_band_match_rate', 0.0)}",
        "",
        "## Cases",
    ]

    for comparison in comparisons:
        if not isinstance(comparison, dict):
            continue
        candidate_id = comparison.get("candidate_id", "unknown")
        model = comparison.get("model", {}) if isinstance(comparison.get("model"), dict) else {}
        human = comparison.get("human_review", {}) if isinstance(comparison.get("human_review"), dict) else {}
        lines.extend(
            [
                "",
                f"### {candidate_id}",
                f"- Model recommendation: {model.get('recommendation', 'n/a')}",
                f"- Human recommendation: {human.get('recommendation', 'n/a')}",
                f"- Model shortlist band: {model.get('shortlist_band', False)}",
                f"- Human shortlist band: {human.get('shortlist_band', False)}",
            ]
        )
        notes = comparison.get("notes")
        if notes:
            lines.append(f"- Notes: {notes}")

    return "\n".join(lines)
