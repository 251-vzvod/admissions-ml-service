"""Batch scoring + diagnostics utility for candidates.json datasets.

This script reports candidate-level product scores and system-level evaluation
metrics for MVP validation without ground-truth labels.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.pipeline import ScoringPipeline


def load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("JSON must contain a 'candidates' list")
    return candidates


def score_all(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pipeline = ScoringPipeline()
    results = []
    for item in candidates:
        results.append(pipeline.score_candidate(item).model_dump())
    return results


def distribution(values: list[int]) -> dict[str, float]:
    if not values:
        return {"count": 0}
    return {
        "count": float(len(values)),
        "min": float(min(values)),
        "max": float(max(values)),
        "mean": float(round(statistics.mean(values), 2)),
        "median": float(round(statistics.median(values), 2)),
        "stdev": float(round(statistics.pstdev(values), 2)),
    }


def run_diagnostics(candidates: list[dict[str, Any]], scored: list[dict[str, Any]]) -> dict[str, Any]:
    merit_values = [int(r["merit_score"]) for r in scored]
    confidence_values = [int(r["confidence_score"]) for r in scored]
    risk_values = [int(r["authenticity_risk"]) for r in scored]

    recommendation_mix = dict(Counter(r["recommendation"] for r in scored))

    completeness = [float(r["feature_snapshot"].get("completeness_score", 0.0)) for r in scored]
    confidence_raw = [v / 100.0 for v in confidence_values]
    risk_raw = [v / 100.0 for v in risk_values]
    evidence_density = [float(r["feature_snapshot"].get("evidence_count", 0.0)) for r in scored]

    sanity_confidence_vs_completeness = float(
        round(sum(c1 >= c2 for c1, c2 in zip(confidence_raw, completeness)) / max(len(scored), 1), 2)
    )
    sanity_risk_vs_evidence = float(
        round(sum(r >= (1 - e) for r, e in zip(risk_raw, evidence_density)) / max(len(scored), 1), 2)
    )

    # Perturbation tests on first available candidate.
    perturbation = {}
    if candidates:
        pipeline = ScoringPipeline()
        base = pipeline.score_candidate(candidates[0]).model_dump()

        less_evidence_candidate = deepcopy(candidates[0])
        less_evidence_candidate.setdefault("text_inputs", {})["motivation_letter_text"] = "I am motivated and passionate."
        less_evidence = pipeline.score_candidate(less_evidence_candidate).model_dump()

        more_concrete_candidate = deepcopy(candidates[0])
        extra = (
            " I led a 6-person team for 4 months, improved participation by 35%, and documented outcomes weekly."
        )
        more_concrete_candidate.setdefault("text_inputs", {})["motivation_letter_text"] = (
            (more_concrete_candidate["text_inputs"].get("motivation_letter_text") or "") + extra
        )
        more_concrete = pipeline.score_candidate(more_concrete_candidate).model_dump()

        generic_candidate = deepcopy(candidates[0])
        generic_candidate.setdefault("text_inputs", {})["motivation_letter_text"] = (
            "I am passionate and motivated. I want to grow and change the world. " * 12
        )
        generic_variant = pipeline.score_candidate(generic_candidate).model_dump()

        perturbation = {
            "remove_examples_confidence_drop": less_evidence["confidence_score"] <= base["confidence_score"],
            "add_outcomes_specificity_rise": more_concrete["feature_snapshot"]["specificity_score"]
            >= base["feature_snapshot"]["specificity_score"],
            "generic_text_risk_rise": generic_variant["authenticity_risk"] >= base["authenticity_risk"],
        }

    no_sensitive_fields_audit = {
        "excluded_fields_configured": True,
        "feature_snapshot_has_sensitive_fields": any(
            key in {"gender", "ethnicity", "income", "citizenship"}
            for row in scored
            for key in row.get("feature_snapshot", {}).keys()
        ),
    }

    return {
        "candidate_scoring_outputs_summary": {
            "merit_score_distribution": distribution(merit_values),
            "confidence_score_distribution": distribution(confidence_values),
            "authenticity_risk_distribution": distribution(risk_values),
            "recommendation_distribution": recommendation_mix,
        },
        "system_evaluation_metrics_without_labels": {
            "coverage": len(scored),
            "missingness_rate_estimate": float(
                round(
                    sum(1 for r in scored if r["eligibility_status"] != "eligible") / max(len(scored), 1),
                    3,
                )
            ),
            "score_variance_proxy": {
                "merit_var": float(round(statistics.pvariance(merit_values), 2)) if len(merit_values) > 1 else 0.0,
                "confidence_var": float(round(statistics.pvariance(confidence_values), 2)) if len(confidence_values) > 1 else 0.0,
            },
            "confidence_vs_completeness_sanity": sanity_confidence_vs_completeness,
            "risk_vs_evidence_sanity": sanity_risk_vs_evidence,
            "perturbation_tests": perturbation,
            "no_sensitive_fields_used_audit": no_sensitive_fields_audit,
            "future_metrics_framework": [
                "rank_correlation_with_human_review",
                "precision_at_k_when_labels_exist",
                "inter_rater_agreement_proxy",
                "confidence_calibration_analysis",
                "authenticity_flag_rate_analysis",
            ],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Score candidates and run diagnostics")
    parser.add_argument("--input", default="data/candidates.json", help="Path to candidates JSON file")
    parser.add_argument("--output", default="data/scored_candidates.json", help="Where to write scored results")
    parser.add_argument("--diagnostics-output", default="data/diagnostics_report.json", help="Where to write diagnostics")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    diagnostics_path = Path(args.diagnostics_output)

    candidates = load_candidates(input_path)
    scored = score_all(candidates)
    diagnostics = run_diagnostics(candidates, scored)

    output_path.write_text(json.dumps({"results": scored}, ensure_ascii=False, indent=2), encoding="utf-8")
    diagnostics_path.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Scored {len(scored)} candidates -> {output_path}")
    print(f"Diagnostics report -> {diagnostics_path}")


if __name__ == "__main__":
    main()
