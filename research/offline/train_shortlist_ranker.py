"""Train an offline shortlist ranker artifact from annotated candidate data."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.offline_ranker import RANKER_ARTIFACT_PATH, build_offline_ranker_feature_map
from app.services.pipeline import ScoringPipeline
from research.annotation_eval import CandidateAnnotation, build_label_evaluation, load_annotations


FEATURE_KEYS = [
    "shortlist_priority_score",
    "hidden_potential_score",
    "trajectory_score",
    "evidence_coverage_score",
    "merit_score",
    "confidence_score",
    "authenticity_risk_neg",
    "community_values_axis",
    "leadership_axis",
    "potential_axis",
    "trust_axis",
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


def _vector_from_result(result: object) -> list[float]:
    feature_map = build_offline_ranker_feature_map(result)
    return [float(feature_map.get(key, 0.0)) for key in FEATURE_KEYS]


def _dot(weights: list[float], features: list[float]) -> float:
    return sum(w * x for w, x in zip(weights, features))


def _sigmoid(value: float) -> float:
    if value >= 0:
        exp = math.exp(-value)
        return 1.0 / (1.0 + exp)
    exp = math.exp(value)
    return exp / (1.0 + exp)


def _build_examples(candidate_rows: list[tuple[str, list[float], CandidateAnnotation]]) -> list[tuple[list[float], int]]:
    examples: list[tuple[list[float], int]] = []
    for i in range(len(candidate_rows)):
        _left_id, left_vec, left_ann = candidate_rows[i]
        for j in range(i + 1, len(candidate_rows)):
            _right_id, right_vec, right_ann = candidate_rows[j]
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


def train_pairwise_ranker(examples: list[tuple[list[float], int]], epochs: int = 40, lr: float = 0.20) -> tuple[list[float], float]:
    weights = [0.0] * len(FEATURE_KEYS)
    bias = 0.0
    rng = random.Random(42)
    rows = list(examples)
    for epoch in range(epochs):
        rng.shuffle(rows)
        step = lr / (1.0 + (epoch * 0.08))
        for features, label in rows:
            pred = _sigmoid(_dot(weights, features) + bias)
            error = pred - float(label)
            for idx, value in enumerate(features):
                weights[idx] -= step * error * value
            bias -= step * error
    return weights, bias


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
        rows.append(score_row)
        vectors[cid] = _vector_from_result(type("ResultStub", (), score_row)())
        families[cid] = _family_id(candidate)
    return rows, vectors, families


def build_train_test_split(families: dict[str, str]) -> tuple[set[str], set[str]]:
    family_ids = sorted(set(families.values()))
    test_families = {family_id for idx, family_id in enumerate(family_ids) if idx % 5 == 0}
    train_ids = {cid for cid, family in families.items() if family not in test_families}
    test_ids = {cid for cid, family in families.items() if family in test_families}
    return train_ids, test_ids


def build_report(candidates: list[dict[str, Any]], annotations: dict[str, CandidateAnnotation]) -> dict[str, Any]:
    rows, vectors, families = build_rank_rows(candidates, annotations)
    row_by_id = {row["candidate_id"]: row for row in rows}

    train_ids, test_ids = build_train_test_split(families)
    train_rows = [(cid, vectors[cid], annotations[cid]) for cid in train_ids if cid in vectors]
    test_rows = [(cid, vectors[cid], annotations[cid]) for cid in test_ids if cid in vectors]

    examples = _build_examples(train_rows)
    weights, bias = train_pairwise_ranker(examples)

    baseline_test = [row_by_id[cid] for cid in test_ids if cid in row_by_id]
    learned_test: list[dict[str, Any]] = []
    for cid, vector, _annotation in test_rows:
        learned_row = dict(row_by_id[cid])
        learned_score = _dot(weights, vector) + bias
        learned_display = max(0, min(100, int(round((learned_score + 1.0) * 50.0))))
        learned_row["merit_score"] = learned_display
        learned_row["shortlist_priority_score"] = learned_display
        learned_test.append(learned_row)

    baseline_eval = build_label_evaluation(baseline_test, annotations)
    learned_eval = build_label_evaluation(learned_test, annotations)
    feature_weights = {feature: round(weight, 6) for feature, weight in zip(FEATURE_KEYS, weights)}

    return {
        "artifact": {
            "version": "offline-shortlist-ranker-v1",
            "bias": round(bias, 6),
            "feature_weights": feature_weights,
        },
        "meta": {
            "train_candidate_count": len(train_rows),
            "test_candidate_count": len(test_rows),
            "train_pair_count": len(examples),
            "split": "family_aware_deterministic_80_20",
        },
        "baseline_test_eval": baseline_eval,
        "learned_pairwise_eval": learned_eval,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train offline shortlist ranker artifact")
    parser.add_argument("--input", default="research/data/candidates_expanded_v1.json")
    parser.add_argument("--annotations", default="research/data/final_hackathon_annotations_v1.json")
    parser.add_argument("--output-artifact", default=str(RANKER_ARTIFACT_PATH))
    parser.add_argument("--output-report", default="research/reports/offline_shortlist_ranker_report.json")
    args = parser.parse_args()

    candidates = load_candidates(Path(args.input))
    annotations = load_annotations(args.annotations)
    report = build_report(candidates, annotations)

    artifact_path = Path(args.output_artifact)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(report["artifact"], ensure_ascii=False, indent=2), encoding="utf-8")

    report_path = Path(args.output_report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Offline ranker artifact -> {artifact_path}")
    print(f"Offline ranker report -> {report_path}")


if __name__ == "__main__":
    main()
