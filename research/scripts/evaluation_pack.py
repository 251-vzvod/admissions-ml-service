"""Offline evaluation pack for scoring quality, reliability, fairness, and baseline comparison.

This script does not alter runtime architecture. It runs:
1) Hybrid pipeline scoring (current production flow)
2) Deterministic baseline replay (offline comparator only)
3) Reliability metrics
4) Fairness audit summaries
5) Config snapshot export for reproducibility
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import CONFIG, build_scoring_config_snapshot
from research.annotation_eval import build_label_evaluation, load_annotations
from app.services.authenticity import estimate_authenticity_risk
from app.services.eligibility import evaluate_eligibility
from app.services.pipeline import ScoringPipeline
from app.services.preprocessing import preprocess_text_inputs
from app.services.privacy import merit_safe_projection
from app.services.recommendation import map_recommendation
from app.services.scoring import compute_scores
from app.services.structured_features import extract_structured_features
from app.services.text_features import extract_text_features
from app.utils.math_utils import to_display_score


def load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("JSON must contain a 'candidates' list")
    return candidates


def score_hybrid(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    pipeline = ScoringPipeline()
    return [pipeline.score_candidate(item, enable_llm_explainability=False).model_dump() for item in candidates]


def score_deterministic_baseline(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for candidate_payload in candidates:
        projected, excluded_hits = merit_safe_projection(candidate_payload)

        text_inputs = projected.get("text_inputs") if isinstance(projected, dict) else {}
        if not isinstance(text_inputs, dict):
            text_inputs = {}

        bundle = preprocess_text_inputs(text_inputs=text_inputs)
        eligibility = evaluate_eligibility(
            candidate_id=str(projected.get("candidate_id", "")),
            consent=projected.get("consent"),
            bundle=bundle,
        )

        if eligibility.status in {"invalid", "incomplete_application"}:
            recommendation = "invalid" if eligibility.status == "invalid" else "incomplete_application"
            results.append(
                {
                    "candidate_id": str(projected.get("candidate_id", "")),
                    "eligibility_status": eligibility.status,
                    "merit_score": 0,
                    "confidence_score": 0,
                    "authenticity_risk": 0,
                    "recommendation": recommendation,
                    "review_flags": ["eligibility_gate"],
                    "llm_metadata": None,
                    "semantic_rubric_scores": {},
                    "feature_snapshot": {
                        "completeness_score": bundle.stats.get("non_empty_text_sources", 0) / 3.0,
                        "excluded_sensitive_fields_count": len(excluded_hits),
                    },
                }
            )
            continue

        structured = extract_structured_features(
            structured_data=projected.get("structured_data") if isinstance(projected.get("structured_data"), dict) else None,
            behavioral_signals=projected.get("behavioral_signals") if isinstance(projected.get("behavioral_signals"), dict) else None,
            bundle=bundle,
        )
        text = extract_text_features(bundle=bundle, structured=structured.features)

        merged: dict[str, float | bool] = {}
        merged.update(structured.features)
        merged.update(text.features)

        auth = estimate_authenticity_risk(features=merged, diagnostics=text.diagnostics)
        merged["authenticity_risk_raw"] = auth.authenticity_risk_raw

        scoring = compute_scores(
            feature_map=merged,
            authenticity_risk_raw=auth.authenticity_risk_raw,
            use_semantic_layer=False,
        )
        recommendation_result = map_recommendation(
            eligibility_status=eligibility.status,
            merit_raw=scoring.merit_raw,
            confidence_raw=scoring.confidence_raw,
            authenticity_risk_raw=scoring.authenticity_risk_raw,
            feature_map=merged,
            prior_flags=auth.review_flags,
        )

        breakdown = {k: to_display_score(v) for k, v in scoring.merit_breakdown_raw.items()}
        results.append(
            {
                "candidate_id": str(projected.get("candidate_id", "")),
                "eligibility_status": eligibility.status,
                "merit_score": scoring.merit_score,
                "confidence_score": scoring.confidence_score,
                "authenticity_risk": scoring.authenticity_risk,
                "recommendation": recommendation_result.recommendation,
                "review_flags": recommendation_result.review_flags,
                "llm_metadata": None,
                "semantic_rubric_scores": {},
                "merit_breakdown": breakdown,
                "feature_snapshot": {
                    "motivation_clarity": round(float(merged.get("motivation_clarity", 0.0)), 4),
                    "initiative": round(float(merged.get("initiative", 0.0)), 4),
                    "leadership_impact": round(float(merged.get("leadership_impact", 0.0)), 4),
                    "growth_trajectory": round(float(merged.get("growth_trajectory", 0.0)), 4),
                    "resilience": round(float(merged.get("resilience", 0.0)), 4),
                    "program_fit": round(float(merged.get("program_fit", 0.0)), 4),
                    "evidence_richness": round(float(merged.get("evidence_richness", 0.0)), 4),
                    "specificity_score": round(float(merged.get("specificity_score", 0.0)), 4),
                    "evidence_count": round(float(merged.get("evidence_count", 0.0)), 4),
                    "consistency_score": round(float(merged.get("consistency_score", 0.0)), 4),
                    "completeness_score": round(float(merged.get("completeness_score", 0.0)), 4),
                    "genericness_score": round(float(merged.get("genericness_score", 0.0)), 4),
                    "contradiction_flag": bool(merged.get("contradiction_flag", False)),
                    "polished_but_empty_score": round(float(merged.get("polished_but_empty_score", 0.0)), 4),
                    "cross_section_mismatch_score": round(float(merged.get("cross_section_mismatch_score", 0.0)), 4),
                    "authenticity_risk_raw": round(float(merged.get("authenticity_risk_raw", 0.0)), 4),
                    "excluded_sensitive_fields_count": len(excluded_hits),
                },
            }
        )

    return results


def _mean(items: list[int]) -> float:
    return round(float(statistics.mean(items)), 3) if items else 0.0


def build_baseline_comparison(hybrid: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> dict[str, Any]:
    by_id_hybrid = {row["candidate_id"]: row for row in hybrid}
    by_id_baseline = {row["candidate_id"]: row for row in baseline}

    common_ids = sorted(set(by_id_hybrid).intersection(by_id_baseline))
    if not common_ids:
        return {"status": "no_overlap"}

    merit_delta = []
    confidence_delta = []
    risk_delta = []
    recommendation_agreement = 0

    for cid in common_ids:
        h = by_id_hybrid[cid]
        b = by_id_baseline[cid]
        merit_delta.append(int(h["merit_score"]) - int(b["merit_score"]))
        confidence_delta.append(int(h["confidence_score"]) - int(b["confidence_score"]))
        risk_delta.append(int(h["authenticity_risk"]) - int(b["authenticity_risk"]))
        if h.get("recommendation") == b.get("recommendation"):
            recommendation_agreement += 1

    return {
        "candidate_count": len(common_ids),
        "hybrid_avg": {
            "merit_score": _mean([int(by_id_hybrid[c]["merit_score"]) for c in common_ids]),
            "confidence_score": _mean([int(by_id_hybrid[c]["confidence_score"]) for c in common_ids]),
            "authenticity_risk": _mean([int(by_id_hybrid[c]["authenticity_risk"]) for c in common_ids]),
            "recommendation_distribution": dict(Counter(by_id_hybrid[c]["recommendation"] for c in common_ids)),
        },
        "deterministic_baseline_avg": {
            "merit_score": _mean([int(by_id_baseline[c]["merit_score"]) for c in common_ids]),
            "confidence_score": _mean([int(by_id_baseline[c]["confidence_score"]) for c in common_ids]),
            "authenticity_risk": _mean([int(by_id_baseline[c]["authenticity_risk"]) for c in common_ids]),
            "recommendation_distribution": dict(Counter(by_id_baseline[c]["recommendation"] for c in common_ids)),
        },
        "delta_hybrid_minus_baseline": {
            "merit_score_mean_delta": _mean(merit_delta),
            "confidence_score_mean_delta": _mean(confidence_delta),
            "authenticity_risk_mean_delta": _mean(risk_delta),
        },
        "recommendation_agreement_rate": round(recommendation_agreement / len(common_ids), 3),
    }


def build_reliability_report(hybrid: list[dict[str, Any]]) -> dict[str, Any]:
    total = max(len(hybrid), 1)
    if hybrid and all(row.get("llm_metadata") is None for row in hybrid):
        return {
            "coverage": len(hybrid),
            "mode": "llm_explainability_disabled_for_offline_eval",
            "llm_extraction_success_rate": None,
            "llm_fallback_rate": None,
            "fallback_reason_distribution": {},
        }
    fallback_rows = [
        row
        for row in hybrid
        if isinstance(row.get("llm_metadata"), dict) and "fallback_reason" in row["llm_metadata"]
    ]
    fallback_reasons = Counter(row["llm_metadata"].get("fallback_reason", "unknown") for row in fallback_rows)
    return {
        "coverage": len(hybrid),
        "llm_extraction_success_rate": round((len(hybrid) - len(fallback_rows)) / total, 3),
        "llm_fallback_rate": round(len(fallback_rows) / total, 3),
        "fallback_reason_distribution": dict(fallback_reasons),
    }


def _word_count_from_candidate(candidate: dict[str, Any]) -> int:
    text_inputs = candidate.get("text_inputs", {})
    if not isinstance(text_inputs, dict):
        return 0

    parts: list[str] = []
    motivation_letter = text_inputs.get("motivation_letter_text")
    if isinstance(motivation_letter, str):
        parts.append(motivation_letter)

    interview_text = text_inputs.get("interview_text")
    if isinstance(interview_text, str):
        parts.append(interview_text)

    qas = text_inputs.get("motivation_questions", [])
    if isinstance(qas, list):
        for qa in qas:
            if isinstance(qa, dict):
                ans = qa.get("answer")
                if isinstance(ans, str):
                    parts.append(ans)

    text = "\n".join(parts)
    return len(text.split())


def _text_length_bucket(words: int) -> str:
    if words < 120:
        return "short"
    if words < 320:
        return "medium"
    return "long"


def build_fairness_audit(candidates: list[dict[str, Any]], hybrid: list[dict[str, Any]]) -> dict[str, Any]:
    by_id_scores = {row["candidate_id"]: row for row in hybrid}

    groups: dict[str, dict[str, list[dict[str, Any]]]] = {
        "language_profile": defaultdict(list),
        "region": defaultdict(list),
        "text_length_bucket": defaultdict(list),
    }

    for candidate in candidates:
        cid = str(candidate.get("candidate_id", ""))
        if cid not in by_id_scores:
            continue

        row = by_id_scores[cid]
        content_profile = candidate.get("content_profile", {})
        if not isinstance(content_profile, dict):
            content_profile = {}

        language = str(content_profile.get("language_profile", "unknown") or "unknown")
        region = str(content_profile.get("city_or_region", "unknown") or "unknown")
        words = _word_count_from_candidate(candidate)
        length_bucket = _text_length_bucket(words)

        groups["language_profile"][language].append(row)
        groups["region"][region].append(row)
        groups["text_length_bucket"][length_bucket].append(row)

    def summarize_group(items: list[dict[str, Any]]) -> dict[str, Any]:
        n = len(items)
        if n == 0:
            return {"count": 0}
        return {
            "count": n,
            "avg_merit_score": _mean([int(x["merit_score"]) for x in items]),
            "avg_confidence_score": _mean([int(x["confidence_score"]) for x in items]),
            "avg_authenticity_risk": _mean([int(x["authenticity_risk"]) for x in items]),
            "recommendation_distribution": dict(Counter(x["recommendation"] for x in items)),
            "manual_review_rate": round(
                sum(1 for x in items if x["recommendation"] in {"manual_review_required", "insufficient_evidence"})
                / n,
                3,
            ),
        }

    fairness = {}
    for axis, axis_groups in groups.items():
        fairness[axis] = {label: summarize_group(rows) for label, rows in sorted(axis_groups.items())}

    fairness["privacy_guardrails"] = {
        "sensitive_fields_in_feature_snapshot": any(
            key in {"gender", "ethnicity", "income", "citizenship", "race", "religion"}
            for row in hybrid
            for key in row.get("feature_snapshot", {}).keys()
        ),
        "excluded_sensitive_fields_count_distribution": dict(
            Counter(int(row.get("feature_snapshot", {}).get("excluded_sensitive_fields_count", 0)) for row in hybrid)
        ),
    }
    return fairness


def build_config_snapshot_with_hash() -> dict[str, Any]:
    snapshot = build_scoring_config_snapshot()
    canonical = json.dumps(snapshot, ensure_ascii=False, sort_keys=True)
    config_hash = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return {
        "config_hash_sha256": config_hash,
        "snapshot": snapshot,
    }


def build_eval_report(
    candidates: list[dict[str, Any]],
    hybrid: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    label_evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    report = {
        "meta": {
            "candidate_count": len(candidates),
            "scoring_version": CONFIG.scoring_version,
            "scoring_config_version": CONFIG.scoring_config_version,
            "weight_experiment_protocol_version": CONFIG.weight_experiment_protocol_version,
            "extraction_strategy": "deterministic_features_plus_semantic_rubrics_plus_optional_llm_explainability",
            "ranking_strategy": "deterministic_baseline_plus_semantic_rubric_layer",
        },
        "llm_reliability": build_reliability_report(hybrid),
        "baseline_comparison": build_baseline_comparison(hybrid, baseline),
        "fairness_audit": build_fairness_audit(candidates, hybrid),
        "notes": [
            "Current LLM layer is explainability-only and does not change numeric scoring.",
            "Current hybrid pipeline adds a lightweight semantic rubric layer on top of the deterministic baseline.",
            "Deterministic baseline comparison is offline-only and should be replaced by labeled ranking comparison for model iterations.",
            "Fairness audit is descriptive and should be complemented with committee review.",
        ],
    }
    if label_evaluation is not None:
        report["label_evaluation"] = label_evaluation
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run offline evaluation pack")
    parser.add_argument("--input", default="data/archive/candidates.json", help="Path to candidates JSON")
    parser.add_argument(
        "--output-dir",
        default="research/reports/evaluation_pack",
        help="Output directory for evaluation artifacts",
    )
    parser.add_argument(
        "--annotations",
        default=None,
        help="Optional path to committee annotation file for ranking evaluation",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(input_path)
    hybrid = score_hybrid(candidates)
    baseline = score_deterministic_baseline(candidates)

    baseline_comparison = build_baseline_comparison(hybrid, baseline)
    fairness_audit = build_fairness_audit(candidates, hybrid)
    config_snapshot = build_config_snapshot_with_hash()
    label_evaluation = None
    if args.annotations:
        annotations = load_annotations(args.annotations)
        label_evaluation = build_label_evaluation(hybrid, annotations)
        (output_dir / "label_evaluation.json").write_text(
            json.dumps(label_evaluation, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    report = build_eval_report(candidates, hybrid, baseline, label_evaluation=label_evaluation)

    (output_dir / "hybrid_scored_results.json").write_text(
        json.dumps({"results": hybrid}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "deterministic_baseline_results.json").write_text(
        json.dumps({"results": baseline}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "baseline_comparison.json").write_text(
        json.dumps(baseline_comparison, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "fairness_audit.json").write_text(
        json.dumps(fairness_audit, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "scoring_config_snapshot.json").write_text(
        json.dumps(config_snapshot, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (output_dir / "evaluation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Evaluation pack generated in: {output_dir}")


if __name__ == "__main__":
    main()
