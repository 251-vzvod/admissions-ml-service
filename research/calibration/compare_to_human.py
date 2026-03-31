"""Compare small adjudicated calibration set against current model policy."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from app.services.pipeline import ScoringPipeline


POLICY_KEYS = (
    "shortlist_band",
    "hidden_potential_band",
    "support_needed_band",
    "authenticity_review_band",
)


@dataclass(slots=True)
class CalibrationCase:
    candidate_id: str
    candidate_payload: dict[str, Any]
    human_review: dict[str, Any]


def load_calibration_cases(path: str | Path) -> list[CalibrationCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    items = payload.get("cases", [])
    if not isinstance(items, list):
        raise ValueError("calibration file must contain a top-level 'cases' list")

    cases: list[CalibrationCase] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        candidate_id = str(item.get("candidate_id", "")).strip()
        candidate_payload = item.get("candidate_payload")
        human_review = item.get("human_review")
        if not candidate_id or not isinstance(candidate_payload, dict) or not isinstance(human_review, dict):
            continue
        cases.append(
            CalibrationCase(
                candidate_id=candidate_id,
                candidate_payload=candidate_payload,
                human_review=human_review,
            )
        )
    return cases


def compare_cases(cases: list[CalibrationCase]) -> dict[str, Any]:
    pipeline = ScoringPipeline()
    comparisons: list[dict[str, Any]] = []

    recommendation_matches = 0
    full_policy_matches = 0
    band_matches = {key: 0 for key in POLICY_KEYS}

    for case in cases:
        response = pipeline.score_candidate(case.candidate_payload)
        trace = pipeline.score_candidate_trace(case.candidate_payload)
        model_policy = trace.get("policy", {}) if isinstance(trace.get("policy"), dict) else {}

        human_recommendation = str(case.human_review.get("recommendation", "")).strip()
        model_recommendation = str(response.recommendation)
        recommendation_match = human_recommendation == model_recommendation
        if recommendation_match:
            recommendation_matches += 1

        band_comparison: dict[str, dict[str, Any]] = {}
        local_full_policy_match = True
        for key in POLICY_KEYS:
            human_value = bool(case.human_review.get(key, False))
            model_value = bool(model_policy.get(key, False))
            match = human_value == model_value
            if match:
                band_matches[key] += 1
            else:
                local_full_policy_match = False
            band_comparison[key] = {
                "human": human_value,
                "model": model_value,
                "match": match,
            }

        if local_full_policy_match and recommendation_match:
            full_policy_matches += 1

        comparisons.append(
            {
                "candidate_id": case.candidate_id,
                "recommendation": {
                    "human": human_recommendation,
                    "model": model_recommendation,
                    "match": recommendation_match,
                },
                "bands": band_comparison,
                "model_scores": {
                    "merit_score": response.merit_score,
                    "confidence_score": response.confidence_score,
                    "authenticity_risk": response.authenticity_risk,
                    "hidden_potential_score": response.hidden_potential_score,
                    "support_needed_score": response.support_needed_score,
                    "shortlist_priority_score": response.shortlist_priority_score,
                    "evidence_coverage_score": response.evidence_coverage_score,
                    "trajectory_score": response.trajectory_score,
                },
                "committee_cohorts": list(response.committee_cohorts),
                "human_notes": str(case.human_review.get("notes", "")).strip(),
            }
        )

    total = len(comparisons)
    summary = {
        "case_count": total,
        "recommendation_exact_match_rate": _safe_rate(recommendation_matches, total),
        "full_policy_match_rate": _safe_rate(full_policy_matches, total),
        "band_match_rate": {key: _safe_rate(value, total) for key, value in band_matches.items()},
        "mismatch_count": sum(
            1
            for item in comparisons
            if not item["recommendation"]["match"] or not all(band["match"] for band in item["bands"].values())
        ),
    }
    return {
        "summary": summary,
        "comparisons": comparisons,
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    summary = report.get("summary", {})
    comparisons = report.get("comparisons", [])
    mismatch_rows = [
        item
        for item in comparisons
        if not item["recommendation"]["match"] or not all(band["match"] for band in item["bands"].values())
    ]

    lines = [
        "# Calibration Report",
        "",
        "## Summary",
        f"- cases: `{summary.get('case_count', 0)}`",
        f"- recommendation exact match: `{summary.get('recommendation_exact_match_rate', 0.0)}`",
        f"- full policy match: `{summary.get('full_policy_match_rate', 0.0)}`",
        f"- mismatch count: `{summary.get('mismatch_count', 0)}`",
        f"- shortlist band match: `{summary.get('band_match_rate', {}).get('shortlist_band', 0.0)}`",
        f"- hidden potential band match: `{summary.get('band_match_rate', {}).get('hidden_potential_band', 0.0)}`",
        f"- support needed band match: `{summary.get('band_match_rate', {}).get('support_needed_band', 0.0)}`",
        f"- authenticity review band match: `{summary.get('band_match_rate', {}).get('authenticity_review_band', 0.0)}`",
        "",
        "## Mismatches",
    ]

    if not mismatch_rows:
        lines.append("- no mismatches")
        return "\n".join(lines) + "\n"

    for item in mismatch_rows:
        lines.extend(
            [
                f"### {item['candidate_id']}",
                f"- recommendation: human=`{item['recommendation']['human']}` model=`{item['recommendation']['model']}`",
                f"- shortlist band: human=`{item['bands']['shortlist_band']['human']}` model=`{item['bands']['shortlist_band']['model']}`",
                f"- hidden potential band: human=`{item['bands']['hidden_potential_band']['human']}` model=`{item['bands']['hidden_potential_band']['model']}`",
                f"- support needed band: human=`{item['bands']['support_needed_band']['human']}` model=`{item['bands']['support_needed_band']['model']}`",
                f"- authenticity review band: human=`{item['bands']['authenticity_review_band']['human']}` model=`{item['bands']['authenticity_review_band']['model']}`",
                f"- model scores: merit=`{item['model_scores']['merit_score']}`, confidence=`{item['model_scores']['confidence_score']}`, authenticity_risk=`{item['model_scores']['authenticity_risk']}`, shortlist_priority=`{item['model_scores']['shortlist_priority_score']}`",
                f"- cohorts: `{', '.join(item['committee_cohorts'])}`",
                f"- human notes: {item['human_notes'] or 'n/a'}",
                "",
            ]
        )

    return "\n".join(lines)


def _safe_rate(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(numerator / denominator, 4)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare adjudicated calibration cases against current model policy")
    parser.add_argument("--input", default="research/calibration/adjudication_template.json")
    parser.add_argument("--output-json", default="research/reports/calibration_report.json")
    parser.add_argument("--output-md", default="research/reports/calibration_report.md")
    args = parser.parse_args()

    cases = load_calibration_cases(args.input)
    report = compare_cases(cases)

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)

    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(build_markdown_report(report), encoding="utf-8")

    print(f"cases={report['summary']['case_count']} mismatches={report['summary']['mismatch_count']}")
    print(f"wrote_json={output_json}")
    print(f"wrote_md={output_md}")


if __name__ == "__main__":
    main()
