"""Experimental learned pairwise ranker over transparent shortlist features.

This is an offline experiment. It does not change runtime scoring.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.annotation_eval import CandidateAnnotation, build_label_evaluation, load_annotations
from app.services.pipeline import ScoringPipeline


FEATURE_KEYS = [
    "merit_score",
    "confidence_score",
    "authenticity_risk_neg",
    "hidden_potential_score",
    "support_needed_score",
    "shortlist_priority_score",
    "evidence_coverage_score",
    "trajectory_score",
    "leadership_potential_semantic",
    "growth_trajectory_semantic",
    "motivation_authenticity_semantic",
    "authenticity_groundedness_semantic",
    "hidden_potential_semantic",
    "evidence_count_text",
    "specificity_score_text",
    "consistency_score_text",
    "genericness_score_neg_text",
    "polished_but_empty_neg_text",
]


def load_candidates(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    candidates = payload.get("candidates", [])
    if not isinstance(candidates, list):
        raise ValueError("JSON must contain a 'candidates' list")
    return candidates


def _family_id(candidate: dict[str, Any]) -> str:
    metadata = candidate.get("metadata", {})
    if isinstance(metadata, dict):
        derived = metadata.get("derived_from_candidate_id")
        if isinstance(derived, str) and derived.strip():
            return derived.strip()
    return str(candidate.get("candidate_id", "")).strip()


def _feature_vector(score_row: dict[str, Any], trace_row: dict[str, Any]) -> list[float]:
    semantic = score_row.get("semantic_rubric_scores", {})
    text_features = trace_row.get("text_features", {})
    return [
        float(score_row.get("merit_score", 0)) / 100.0,
        float(score_row.get("confidence_score", 0)) / 100.0,
        1.0 - (float(score_row.get("authenticity_risk", 0)) / 100.0),
        float(score_row.get("hidden_potential_score", 0)) / 100.0,
        float(score_row.get("support_needed_score", 0)) / 100.0,
        float(score_row.get("shortlist_priority_score", 0)) / 100.0,
        float(score_row.get("evidence_coverage_score", 0)) / 100.0,
        float(score_row.get("trajectory_score", 0)) / 100.0,
        float(semantic.get("leadership_potential", 0)) / 100.0,
        float(semantic.get("growth_trajectory", 0)) / 100.0,
        float(semantic.get("motivation_authenticity", 0)) / 100.0,
        float(semantic.get("authenticity_groundedness", 0)) / 100.0,
        float(semantic.get("hidden_potential", 0)) / 100.0,
        float(text_features.get("evidence_count", 0.0)),
        float(text_features.get("specificity_score", 0.0)),
        float(text_features.get("consistency_score", 0.0)),
        1.0 - float(text_features.get("genericness_score", 0.0)),
        1.0 - float(text_features.get("polished_but_empty_score", 0.0)),
    ]


def _dot(weights: list[float], features: list[float]) -> float:
    return sum(w * x for w, x in zip(weights, features))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp = math.exp(-value)
        return 1.0 / (1.0 + exp)
    exp = math.exp(value)
    return exp / (1.0 + exp)


def _build_examples(
    candidate_rows: list[tuple[str, list[float], CandidateAnnotation]],
) -> list[tuple[list[float], int]]:
    examples: list[tuple[list[float], int]] = []
    for i in range(len(candidate_rows)):
        left_id, left_vec, left_ann = candidate_rows[i]
        for j in range(i + 1, len(candidate_rows)):
            right_id, right_vec, right_ann = candidate_rows[j]
            left_label = left_ann.composite_label
            right_label = right_ann.composite_label
            if left_label == right_label:
                continue
            diff = [l - r for l, r in zip(left_vec, right_vec)]
            if left_label > right_label:
                examples.append((diff, 1))
                examples.append(([-value for value in diff], 0))
            else:
                examples.append((diff, 0))
                examples.append(([-value for value in diff], 1))
    return examples


def train_pairwise_ranker(examples: list[tuple[list[float], int]], epochs: int = 30, lr: float = 0.25) -> list[float]:
    weights = [0.0] * len(FEATURE_KEYS)
    rng = random.Random(42)
    rows = list(examples)
    for epoch in range(epochs):
        rng.shuffle(rows)
        step = lr / (1.0 + (epoch * 0.08))
        for features, label in rows:
            pred = _sigmoid(_dot(weights, features))
            error = pred - float(label)
            for idx, value in enumerate(features):
                weights[idx] -= step * error * value
    return weights


def build_rank_rows(
    candidates: list[dict[str, Any]],
    annotations: dict[str, CandidateAnnotation],
) -> tuple[list[dict[str, Any]], dict[str, list[float]], dict[str, str]]:
    pipeline = ScoringPipeline()
    rows: list[dict[str, Any]] = []
    vectors: dict[str, list[float]] = {}
    families: dict[str, str] = {}
    for candidate in candidates:
        cid = str(candidate.get("candidate_id", "")).strip()
        if cid not in annotations:
            continue
        score_row = pipeline.score_candidate(candidate, enable_llm_explainability=False).model_dump()
        trace_row = pipeline.score_candidate_trace(candidate)
        rows.append(score_row)
        vectors[cid] = _feature_vector(score_row, trace_row)
        families[cid] = _family_id(candidate)
    return rows, vectors, families


def build_train_test_split(families: dict[str, str]) -> tuple[set[str], set[str]]:
    family_ids = sorted(set(families.values()))
    test_families = {family_id for idx, family_id in enumerate(family_ids) if idx % 5 == 0}
    train_ids = {cid for cid, family in families.items() if family not in test_families}
    test_ids = {cid for cid, family in families.items() if family in test_families}
    return train_ids, test_ids


def build_learned_report(
    candidates: list[dict[str, Any]],
    annotations: dict[str, CandidateAnnotation],
) -> dict[str, Any]:
    rows, vectors, families = build_rank_rows(candidates, annotations)
    row_by_id = {row["candidate_id"]: row for row in rows}

    train_ids, test_ids = build_train_test_split(families)

    train_rows = [(cid, vectors[cid], annotations[cid]) for cid in train_ids if cid in vectors]
    test_rows = [(cid, vectors[cid], annotations[cid]) for cid in test_ids if cid in vectors]
    examples = _build_examples(train_rows)
    weights = train_pairwise_ranker(examples)

    baseline_test = [row_by_id[cid] for cid in test_ids if cid in row_by_id]
    learned_test: list[dict[str, Any]] = []
    for cid, vector, _annotation in test_rows:
        learned_row = dict(row_by_id[cid])
        learned_score = _dot(weights, vector)
        learned_display = max(0, min(100, int(round((learned_score + 1.0) * 50.0))))
        learned_row["merit_score"] = learned_display
        learned_row["shortlist_priority_score"] = learned_display
        learned_test.append(learned_row)

    baseline_eval = build_label_evaluation(baseline_test, annotations)
    learned_eval = build_label_evaluation(learned_test, annotations)

    top_weights = sorted(
        zip(FEATURE_KEYS, weights),
        key=lambda item: abs(item[1]),
        reverse=True,
    )[:10]

    return {
        "meta": {
            "train_candidate_count": len(train_rows),
            "test_candidate_count": len(test_rows),
            "train_pair_count": len(examples),
            "split": "family_aware_deterministic_80_20",
        },
        "baseline_test_eval": baseline_eval,
        "learned_pairwise_eval": learned_eval,
        "delta_learned_minus_baseline": {
            "spearman_merit_vs_labels": round(
                float(learned_eval.get("spearman_merit_vs_labels", 0.0))
                - float(baseline_eval.get("spearman_merit_vs_labels", 0.0)),
                4,
            ),
            "pairwise_accuracy": round(
                float(learned_eval.get("pairwise_accuracy", 0.0)) - float(baseline_eval.get("pairwise_accuracy", 0.0)),
                4,
            ),
            "precision_at_k_priority": round(
                float(learned_eval.get("precision_at_k_priority", 0.0))
                - float(baseline_eval.get("precision_at_k_priority", 0.0)),
                4,
            ),
            "hidden_potential_recall_at_k": round(
                float(learned_eval.get("hidden_potential_recall_at_k", 0.0))
                - float(baseline_eval.get("hidden_potential_recall_at_k", 0.0)),
                4,
            ),
        },
        "top_feature_weights": [
            {"feature": feature, "weight": round(weight, 4)} for feature, weight in top_weights
        ],
    }


def build_markdown_report(report: dict[str, Any]) -> str:
    baseline = report["baseline_test_eval"]
    learned = report["learned_pairwise_eval"]
    delta = report["delta_learned_minus_baseline"]
    lines = [
        "# Pairwise Ranker Experiment",
        "",
        "## Scope",
        "",
        f"- train candidates: `{report['meta']['train_candidate_count']}`",
        f"- test candidates: `{report['meta']['test_candidate_count']}`",
        f"- train pairs: `{report['meta']['train_pair_count']}`",
        f"- split: `{report['meta']['split']}`",
        "",
        "## Baseline Test Metrics",
        "",
        f"- spearman: `{baseline.get('spearman_merit_vs_labels', 'n/a')}`",
        f"- pairwise accuracy: `{baseline.get('pairwise_accuracy', 'n/a')}`",
        f"- precision@k priority: `{baseline.get('precision_at_k_priority', 'n/a')}`",
        f"- hidden potential recall@k: `{baseline.get('hidden_potential_recall_at_k', 'n/a')}`",
        "",
        "## Learned Pairwise Test Metrics",
        "",
        f"- spearman: `{learned.get('spearman_merit_vs_labels', 'n/a')}`",
        f"- pairwise accuracy: `{learned.get('pairwise_accuracy', 'n/a')}`",
        f"- precision@k priority: `{learned.get('precision_at_k_priority', 'n/a')}`",
        f"- hidden potential recall@k: `{learned.get('hidden_potential_recall_at_k', 'n/a')}`",
        "",
        "## Delta Learned Minus Baseline",
        "",
        f"- spearman delta: `{delta['spearman_merit_vs_labels']}`",
        f"- pairwise accuracy delta: `{delta['pairwise_accuracy']}`",
        f"- precision@k priority delta: `{delta['precision_at_k_priority']}`",
        f"- hidden potential recall@k delta: `{delta['hidden_potential_recall_at_k']}`",
        "",
        "## Top Feature Weights",
        "",
    ]
    for row in report["top_feature_weights"]:
        lines.append(f"- `{row['feature']}`: `{row['weight']}`")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run experimental pairwise ranker on transparent features")
    parser.add_argument("--input", default="data/candidates_expanded_v1.json")
    parser.add_argument("--annotations", default="data/final_hackathon_annotations_v1.json")
    parser.add_argument("--output-json", default="data/evaluation_pack_final_hackathon_v3/pairwise_ranker_experiment.json")
    parser.add_argument("--output-md", default="docs/pairwise_ranker_experiment.md")
    args = parser.parse_args()

    candidates = load_candidates(Path(args.input))
    annotations = load_annotations(args.annotations)
    report = build_learned_report(candidates, annotations)

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    output_md = Path(args.output_md)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text(build_markdown_report(report), encoding="utf-8")

    print(f"Pairwise ranker experiment JSON -> {output_json}")
    print(f"Pairwise ranker experiment markdown -> {output_md}")


if __name__ == "__main__":
    main()
