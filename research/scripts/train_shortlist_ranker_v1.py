from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import lightgbm as lgb
import numpy as np
from sklearn.metrics import ndcg_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from app.services.offline_ranker import build_offline_ranker_feature_map, score_result_with_offline_ranker
from app.services.pipeline import ScoringPipeline


DATA_ROOT = ROOT / "data" / "ml_workbench"
EXPORTS_DIR = DATA_ROOT / "exports"
LABELS_DIR = DATA_ROOT / "labels"
MODEL_DIR = EXPORTS_DIR / "models" / "shortlist_ranker_v1_training_dataset_v3"

TRAINING_CSV = EXPORTS_DIR / "training_dataset_v3.csv"
PAIRWISE_CSV = LABELS_DIR / "pairwise_labels.csv"
BATCH_JSONL = LABELS_DIR / "batch_shortlist_tasks.jsonl"

SEED_JSONL = DATA_ROOT / "processed" / "english_candidates_api_input_v1.jsonl"
SYNTHETIC_V1_JSONL = DATA_ROOT / "raw" / "generated" / "batch_v1" / "synthetic_batch_v1_api_input.jsonl"
CONTRASTIVE_V2_JSONL = DATA_ROOT / "raw" / "generated" / "contrastive_batch_v2" / "contrastive_batch_v2_api_input.jsonl"
TRANSLATED_V3_JSONL = DATA_ROOT / "raw" / "generated" / "translated_batch_v3" / "translated_batch_v3_api_input.jsonl"
MESSY_V4_JSONL = DATA_ROOT / "raw" / "generated" / "messy_batch_v4" / "messy_batch_v4_api_input.jsonl"
MESSY_V5_JSONL = DATA_ROOT / "raw" / "generated" / "messy_batch_v5" / "messy_batch_v5_api_input.jsonl"
MESSY_V5_EXTENSION_JSONL = DATA_ROOT / "raw" / "generated" / "messy_batch_v5_extension" / "messy_batch_v5_extension_api_input.jsonl"
ORDINARY_V6_JSONL = DATA_ROOT / "raw" / "generated" / "ordinary_batch_v6" / "ordinary_batch_v6_api_input.jsonl"
GAP_FILL_V7_JSONL = DATA_ROOT / "raw" / "generated" / "gap_fill_batch_v7" / "gap_fill_batch_v7_api_input.jsonl"

METRICS_JSON = MODEL_DIR / "metrics_summary.json"
FEATURE_IMPORTANCE_CSV = MODEL_DIR / "feature_importance.csv"
PREDICTIONS_CSV = MODEL_DIR / "candidate_predictions.csv"
MODEL_TXT = MODEL_DIR / "lightgbm_model.txt"

RANDOM_SEED = 20260402
FEATURE_EXCLUSIONS = {"support_needed_score"}
FEATURE_VARIANT_NAME = "drop_support"
LEARNED_BLEND_ALPHA = 0.4
RANKER_KWARGS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 15,
    "min_data_in_leaf": 2,
    "feature_fraction": 1.0,
    "bagging_fraction": 1.0,
    "bagging_freq": 0,
    "random_state": RANDOM_SEED,
    "feature_fraction_seed": RANDOM_SEED,
    "bagging_seed": RANDOM_SEED,
    "data_random_seed": RANDOM_SEED,
    "deterministic": True,
    "force_col_wise": True,
    "num_threads": 1,
    "verbosity": -1,
}


@dataclass(frozen=True)
class CandidateRow:
    candidate_id: str
    split: str
    final_hidden_potential_band: bool


@dataclass(frozen=True)
class RankGroup:
    group_id: str
    split: str
    candidate_ids: list[str]
    relevance: list[float]


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def load_payloads() -> dict[str, dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    for path in [
        SEED_JSONL,
        SYNTHETIC_V1_JSONL,
        CONTRASTIVE_V2_JSONL,
        TRANSLATED_V3_JSONL,
        MESSY_V4_JSONL,
        MESSY_V5_JSONL,
        MESSY_V5_EXTENSION_JSONL,
        ORDINARY_V6_JSONL,
        GAP_FILL_V7_JSONL,
    ]:
        for record in load_jsonl(path):
            candidate_id = record["candidate_id"]
            if candidate_id in payloads:
                raise ValueError(f"Duplicate candidate_id in payload pool: {candidate_id}")
            payloads[candidate_id] = record
    return payloads


def load_training_rows() -> dict[str, CandidateRow]:
    rows: dict[str, CandidateRow] = {}
    with TRAINING_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            candidate_id = row["candidate_id"]
            rows[candidate_id] = CandidateRow(
                candidate_id=candidate_id,
                split=row["split"],
                final_hidden_potential_band=parse_bool(row["final_hidden_potential_band"]),
            )
    return rows


def load_batch_tasks() -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    with BATCH_JSONL.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                tasks.append(json.loads(line))
    return tasks


def load_pairwise_rows() -> list[dict[str, str]]:
    with PAIRWISE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def build_feature_matrix(
    pipeline: ScoringPipeline,
    payloads: dict[str, dict[str, Any]],
) -> tuple[list[str], dict[str, dict[str, float]], dict[str, float]]:
    feature_map_by_candidate: dict[str, dict[str, float]] = {}
    baseline_score_by_candidate: dict[str, float] = {}
    feature_names: list[str] | None = None

    for candidate_id, payload in payloads.items():
        result = pipeline.score_candidate(payload, enable_llm_explainability=False)
        feature_map = build_offline_ranker_feature_map(result)
        feature_map_by_candidate[candidate_id] = feature_map
        baseline_score_by_candidate[candidate_id] = score_result_with_offline_ranker(result)
        if feature_names is None:
            feature_names = list(feature_map.keys())

    if feature_names is None:
        raise ValueError("No payloads loaded for feature extraction.")
    return feature_names, feature_map_by_candidate, baseline_score_by_candidate


def project_rank_groups(
    batch_tasks: list[dict[str, Any]],
    candidate_rows: dict[str, CandidateRow],
    min_group_size: int,
) -> list[RankGroup]:
    groups: list[RankGroup] = []
    for batch in batch_tasks:
        ranked_ids = batch["ranked_candidate_ids"]
        for split in ("train", "validation", "test"):
            split_ids = [candidate_id for candidate_id in ranked_ids if candidate_rows[candidate_id].split == split]
            if len(split_ids) < min_group_size:
                continue
            size = len(split_ids)
            relevance = [float(size - idx) for idx in range(size)]
            groups.append(
                RankGroup(
                    group_id=f"{batch['batch_id']}:{split}",
                    split=split,
                    candidate_ids=split_ids,
                    relevance=relevance,
                )
            )
    return groups


def matrix_for_groups(
    groups: list[RankGroup],
    feature_names: list[str],
    feature_map_by_candidate: dict[str, dict[str, float]],
) -> tuple[np.ndarray, np.ndarray, list[int]]:
    rows: list[list[float]] = []
    targets: list[float] = []
    group_sizes: list[int] = []
    for group in groups:
        group_sizes.append(len(group.candidate_ids))
        for candidate_id, relevance in zip(group.candidate_ids, group.relevance, strict=True):
            fmap = feature_map_by_candidate[candidate_id]
            rows.append([float(fmap[name]) for name in feature_names])
            targets.append(relevance)
    return np.asarray(rows, dtype=np.float32), np.asarray(targets, dtype=np.float32), group_sizes


def score_map_from_model(
    candidate_ids: list[str],
    feature_names: list[str],
    feature_map_by_candidate: dict[str, dict[str, float]],
    model: lgb.LGBMRanker,
) -> dict[str, float]:
    rows = np.asarray(
        [[float(feature_map_by_candidate[candidate_id][name]) for name in feature_names] for candidate_id in candidate_ids],
        dtype=np.float32,
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="X does not have valid feature names, but LGBMRanker was fitted with feature names",
            category=UserWarning,
        )
        predictions = model.predict(rows)
    return {candidate_id: float(score) for candidate_id, score in zip(candidate_ids, predictions, strict=True)}


def mean_ndcg(groups: list[RankGroup], score_map: dict[str, float], k: int, min_group_size: int) -> float | None:
    values: list[float] = []
    for group in groups:
        if len(group.candidate_ids) < min_group_size:
            continue
        y_true = np.asarray([group.relevance], dtype=np.float32)
        y_score = np.asarray([[score_map[candidate_id] for candidate_id in group.candidate_ids]], dtype=np.float32)
        values.append(float(ndcg_score(y_true, y_score, k=min(k, len(group.candidate_ids)))))
    if not values:
        return None
    return float(np.mean(values))


def pairwise_accuracy(
    pairwise_rows: list[dict[str, str]],
    candidate_rows: dict[str, CandidateRow],
    score_map: dict[str, float],
    split: str,
) -> tuple[float | None, int]:
    total = 0
    correct = 0
    for row in pairwise_rows:
        left = row["candidate_id_left"]
        right = row["candidate_id_right"]
        if candidate_rows[left].split != split or candidate_rows[right].split != split:
            continue
        total += 1
        preferred = row["preferred_candidate_id"]
        left_score = score_map[left]
        right_score = score_map[right]
        predicted = left if left_score >= right_score else right
        if predicted == preferred:
            correct += 1
    if total == 0:
        return None, 0
    return correct / total, total


def hidden_potential_recall_at_k(
    groups: list[RankGroup],
    candidate_rows: dict[str, CandidateRow],
    score_map: dict[str, float],
    k: int,
    min_group_size: int,
) -> tuple[float | None, int]:
    recalls: list[float] = []
    evaluated_groups = 0
    for group in groups:
        if len(group.candidate_ids) < min_group_size:
            continue
        positives = [cid for cid in group.candidate_ids if candidate_rows[cid].final_hidden_potential_band]
        if not positives:
            continue
        evaluated_groups += 1
        ranked = sorted(group.candidate_ids, key=lambda cid: score_map[cid], reverse=True)
        top_k = set(ranked[: min(k, len(ranked))])
        recalls.append(len(top_k.intersection(positives)) / len(positives))
    if not recalls:
        return None, evaluated_groups
    return float(np.mean(recalls)), evaluated_groups


def write_feature_importance(model: lgb.LGBMRanker, feature_names: list[str]) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    gain = model.booster_.feature_importance(importance_type="gain")
    split = model.booster_.feature_importance(importance_type="split")
    rows = sorted(
        (
            {
                "feature_name": name,
                "gain_importance": float(g),
                "split_importance": int(s),
            }
            for name, g, s in zip(feature_names, gain, split, strict=True)
        ),
        key=lambda item: item["gain_importance"],
        reverse=True,
    )
    with FEATURE_IMPORTANCE_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature_name", "gain_importance", "split_importance"])
        writer.writeheader()
        writer.writerows(rows)


def write_candidate_predictions(
    candidate_ids: list[str],
    candidate_rows: dict[str, CandidateRow],
    baseline_score_map: dict[str, float],
    raw_model_score_map: dict[str, float],
    learned_score_map: dict[str, float],
) -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "split",
                "baseline_score",
                "raw_model_score",
                "learned_score",
                "raw_score_delta",
                "score_delta",
                "hidden_potential_band",
            ],
        )
        writer.writeheader()
        for candidate_id in sorted(candidate_ids):
            writer.writerow(
                {
                    "candidate_id": candidate_id,
                    "split": candidate_rows[candidate_id].split,
                    "baseline_score": baseline_score_map[candidate_id],
                    "raw_model_score": raw_model_score_map[candidate_id],
                    "learned_score": learned_score_map[candidate_id],
                    "raw_score_delta": raw_model_score_map[candidate_id] - baseline_score_map[candidate_id],
                    "score_delta": learned_score_map[candidate_id] - baseline_score_map[candidate_id],
                    "hidden_potential_band": str(candidate_rows[candidate_id].final_hidden_potential_band).lower(),
                }
            )


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = ScoringPipeline()
    payloads = load_payloads()
    candidate_rows = load_training_rows()
    batch_tasks = load_batch_tasks()
    pairwise_rows = load_pairwise_rows()

    all_feature_names, feature_map_by_candidate, baseline_score_map = build_feature_matrix(pipeline, payloads)
    feature_names = [name for name in all_feature_names if name not in FEATURE_EXCLUSIONS]
    projected_groups_min2 = project_rank_groups(batch_tasks, candidate_rows, min_group_size=2)
    projected_groups_min3 = project_rank_groups(batch_tasks, candidate_rows, min_group_size=3)

    train_groups = [group for group in projected_groups_min3 if group.split == "train"]
    validation_groups = [group for group in projected_groups_min3 if group.split == "validation"]
    test_groups_min2 = [group for group in projected_groups_min2 if group.split == "test"]
    test_groups_min3 = [group for group in projected_groups_min3 if group.split == "test"]

    if not train_groups:
        raise ValueError("No train rank groups available.")
    if not validation_groups:
        raise ValueError("No validation rank groups available.")

    x_train, y_train, train_group_sizes = matrix_for_groups(train_groups, feature_names, feature_map_by_candidate)
    x_val, y_val, val_group_sizes = matrix_for_groups(validation_groups, feature_names, feature_map_by_candidate)

    model = lgb.LGBMRanker(**RANKER_KWARGS)
    model.fit(
        x_train,
        y_train,
        group=train_group_sizes,
        eval_set=[(x_val, y_val)],
        eval_group=[val_group_sizes],
        eval_at=[3, 5],
        callbacks=[lgb.early_stopping(25, verbose=False), lgb.log_evaluation(period=0)],
    )

    candidate_ids = sorted(payloads)
    raw_model_score_map = score_map_from_model(candidate_ids, feature_names, feature_map_by_candidate, model)
    learned_score_map = {
        candidate_id: (LEARNED_BLEND_ALPHA * raw_model_score_map[candidate_id])
        + ((1.0 - LEARNED_BLEND_ALPHA) * baseline_score_map[candidate_id])
        for candidate_id in candidate_ids
    }

    metrics = {
        "group_counts": {
            "train_min3": len(train_groups),
            "validation_min3": len(validation_groups),
            "test_min2": len(test_groups_min2),
            "test_min3": len(test_groups_min3),
        },
        "baseline": {},
        "raw_model": {},
        "learned": {},
        "feature_names": feature_names,
        "all_feature_names": all_feature_names,
        "feature_exclusions": sorted(FEATURE_EXCLUSIONS),
        "feature_variant_name": FEATURE_VARIANT_NAME,
        "learned_blend_alpha": LEARNED_BLEND_ALPHA,
        "best_iteration": int(model.best_iteration_ or model.n_estimators),
    }

    for label, score_map in [("baseline", baseline_score_map), ("raw_model", raw_model_score_map), ("learned", learned_score_map)]:
        validation_ndcg3 = mean_ndcg(validation_groups, score_map, k=3, min_group_size=3)
        validation_ndcg5 = mean_ndcg(validation_groups, score_map, k=5, min_group_size=3)
        train_ndcg3 = mean_ndcg(train_groups, score_map, k=3, min_group_size=3)
        train_ndcg5 = mean_ndcg(train_groups, score_map, k=5, min_group_size=3)
        test_ndcg2 = mean_ndcg(test_groups_min2, score_map, k=2, min_group_size=2)
        test_ndcg3 = mean_ndcg(test_groups_min3, score_map, k=3, min_group_size=3)
        validation_pairwise, validation_pair_count = pairwise_accuracy(pairwise_rows, candidate_rows, score_map, "validation")
        test_pairwise, test_pair_count = pairwise_accuracy(pairwise_rows, candidate_rows, score_map, "test")
        hidden_recall3, hidden_group_count = hidden_potential_recall_at_k(
            validation_groups,
            candidate_rows,
            score_map,
            k=3,
            min_group_size=3,
        )

        metrics[label] = {
            "train_ndcg_at_3": train_ndcg3,
            "train_ndcg_at_5": train_ndcg5,
            "validation_ndcg_at_3": validation_ndcg3,
            "validation_ndcg_at_5": validation_ndcg5,
            "test_ndcg_at_2": test_ndcg2,
            "test_ndcg_at_3": test_ndcg3,
            "validation_pairwise_accuracy": validation_pairwise,
            "validation_pairwise_count": validation_pair_count,
            "test_pairwise_accuracy": test_pairwise,
            "test_pairwise_count": test_pair_count,
            "validation_hidden_potential_recall_at_3": hidden_recall3,
            "validation_hidden_potential_group_count": hidden_group_count,
        }

    metrics["delta_vs_baseline"] = {
        "validation_ndcg_at_3": (
            None
            if metrics["baseline"]["validation_ndcg_at_3"] is None or metrics["learned"]["validation_ndcg_at_3"] is None
            else metrics["learned"]["validation_ndcg_at_3"] - metrics["baseline"]["validation_ndcg_at_3"]
        ),
        "validation_ndcg_at_5": (
            None
            if metrics["baseline"]["validation_ndcg_at_5"] is None or metrics["learned"]["validation_ndcg_at_5"] is None
            else metrics["learned"]["validation_ndcg_at_5"] - metrics["baseline"]["validation_ndcg_at_5"]
        ),
        "validation_pairwise_accuracy": (
            None
            if metrics["baseline"]["validation_pairwise_accuracy"] is None
            or metrics["learned"]["validation_pairwise_accuracy"] is None
            else metrics["learned"]["validation_pairwise_accuracy"] - metrics["baseline"]["validation_pairwise_accuracy"]
        ),
        "validation_hidden_potential_recall_at_3": (
            None
            if metrics["baseline"]["validation_hidden_potential_recall_at_3"] is None
            or metrics["learned"]["validation_hidden_potential_recall_at_3"] is None
            else metrics["learned"]["validation_hidden_potential_recall_at_3"]
            - metrics["baseline"]["validation_hidden_potential_recall_at_3"]
        ),
    }
    metrics["candidate_split_counts"] = dict(Counter(item.split for item in candidate_rows.values()))

    with METRICS_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    model.booster_.save_model(str(MODEL_TXT))
    write_feature_importance(model, feature_names)
    write_candidate_predictions(candidate_ids, candidate_rows, baseline_score_map, raw_model_score_map, learned_score_map)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
