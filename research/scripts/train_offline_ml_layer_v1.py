from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, f1_score, ndcg_score, precision_recall_fscore_support
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from research.scripts.offline_ml_common import (
    DEFAULT_REPR_CONFIG,
    OFFLINE_LAYER_DIR,
    build_feature_row_for_payload,
    build_or_load_candidate_feature_cache,
    feature_names_from_rows,
    load_batch_tasks,
    load_pairwise_rows,
)


RANDOM_SEED = 20260405
MODEL_DIR = OFFLINE_LAYER_DIR
METRICS_JSON = MODEL_DIR / "metrics_summary.json"
SLICE_JSON = MODEL_DIR / "slice_eval_summary.json"
PREDICTIONS_CSV = MODEL_DIR / "candidate_predictions.csv"
REPORT_MD = MODEL_DIR / "training_report.md"
PRIORITY_MODEL_JOBLIB = MODEL_DIR / "priority_model_v1.joblib"
ROUTING_MODEL_JOBLIB = MODEL_DIR / "routing_models_v1.joblib"
PAIRWISE_MODEL_JOBLIB = MODEL_DIR / "pairwise_ranker_v2.joblib"
SPOT_CHECKS_JSON = MODEL_DIR / "manual_spot_checks.json"


@dataclass(frozen=True, slots=True)
class PriorityModelSpec:
    name: str
    factory: Callable[[], Any]


@dataclass(frozen=True, slots=True)
class RoutingModelSpec:
    name: str
    factory: Callable[[], Any]


@dataclass(frozen=True, slots=True)
class PairwiseModelSpec:
    name: str
    c_value: float


def _to_numpy_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> np.ndarray:
    return np.asarray(
        [[float(item["feature_map"].get(name, 0.0)) for name in feature_names] for item in rows],
        dtype=np.float32,
    )


def _split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {
        split: [row for row in rows if row["split"] == split]
        for split in ("train", "validation", "test")
    }


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.zeros(len(values), dtype=np.float64)
    idx = 0
    while idx < len(values):
        start = idx
        current_index = order[idx]
        current_value = values[current_index]
        while idx + 1 < len(values) and values[order[idx + 1]] == current_value:
            idx += 1
        avg_rank = (start + idx) / 2.0
        for pos in range(start, idx + 1):
            ranks[order[pos]] = avg_rank
        idx += 1
    return ranks


def _spearman(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return 0.0
    rank_true = _rankdata(y_true.astype(np.float64))
    rank_pred = _rankdata(y_pred.astype(np.float64))
    left_std = float(np.std(rank_true))
    right_std = float(np.std(rank_pred))
    if left_std == 0.0 or right_std == 0.0:
        return 0.0
    return float(np.corrcoef(rank_true, rank_pred)[0, 1])


def _priority_target(priority_value: int) -> float:
    return float(priority_value - 1) / 3.0


def _priority_to_raw(prediction: np.ndarray) -> np.ndarray:
    clipped = np.clip(prediction, 0.0, 1.0)
    return 1.0 + (clipped * 3.0)


def _priority_metrics(y_true_raw: np.ndarray, y_score_norm: np.ndarray) -> dict[str, float]:
    y_pred_raw = _priority_to_raw(y_score_norm)
    return {
        "spearman": _spearman(y_true_raw, y_pred_raw),
        "mae": float(np.mean(np.abs(y_true_raw - y_pred_raw))),
    }


def _binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | None]:
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    ap = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None
    return {
        "average_precision": ap,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def _best_threshold_for_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.20, 0.80, 13):
        current_f1 = f1_score(y_true, (y_score >= threshold).astype(int), zero_division=0)
        if current_f1 > best_f1:
            best_f1 = float(current_f1)
            best_threshold = float(threshold)
    return best_threshold


def _priority_model_specs() -> list[PriorityModelSpec]:
    return [
        PriorityModelSpec(
            name="ridge_alpha_1_0",
            factory=lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=1.0)),
                ]
            ),
        ),
        PriorityModelSpec(
            name="hist_gbr_depth3",
            factory=lambda: HistGradientBoostingRegressor(
                max_depth=3,
                max_iter=250,
                learning_rate=0.05,
                random_state=RANDOM_SEED,
            ),
        ),
    ]


def _routing_model_specs() -> list[RoutingModelSpec]:
    return [
        RoutingModelSpec(
            name="logreg_balanced",
            factory=lambda: Pipeline(
                [
                    ("scale", StandardScaler()),
                    (
                        "model",
                        LogisticRegression(
                            max_iter=2000,
                            class_weight="balanced",
                            random_state=RANDOM_SEED,
                            solver="liblinear",
                        ),
                    ),
                ]
            ),
        ),
        RoutingModelSpec(
            name="hist_gbc_depth3",
            factory=lambda: HistGradientBoostingClassifier(
                max_depth=3,
                max_iter=250,
                learning_rate=0.05,
                random_state=RANDOM_SEED,
            ),
        ),
    ]


def _pairwise_model_specs() -> list[PairwiseModelSpec]:
    return [
        PairwiseModelSpec(name="pairwise_logreg_c0_5", c_value=0.5),
        PairwiseModelSpec(name="pairwise_logreg_c1_0", c_value=1.0),
        PairwiseModelSpec(name="pairwise_logreg_c2_0", c_value=2.0),
    ]


def _predict_scores(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(x)[:, 1], dtype=np.float32)
    return np.asarray(model.predict(x), dtype=np.float32)


def _build_rank_groups(rows: list[dict[str, Any]], batch_tasks: list[dict[str, Any]], min_group_size: int) -> list[dict[str, Any]]:
    split_by_candidate = {row["candidate_id"]: row["split"] for row in rows}
    groups: list[dict[str, Any]] = []
    for batch in batch_tasks:
        ranked_ids = [candidate_id for candidate_id in batch["ranked_candidate_ids"] if candidate_id in split_by_candidate]
        for split in ("train", "validation", "test"):
            split_ids = [candidate_id for candidate_id in ranked_ids if split_by_candidate[candidate_id] == split]
            if len(split_ids) < min_group_size:
                continue
            relevance = [float(len(split_ids) - idx) for idx in range(len(split_ids))]
            groups.append(
                {
                    "group_id": f"{batch['batch_id']}:{split}",
                    "split": split,
                    "candidate_ids": split_ids,
                    "relevance": relevance,
                }
            )
    return groups


def _mean_ndcg(groups: list[dict[str, Any]], score_map: dict[str, float], k: int, min_group_size: int) -> float | None:
    values: list[float] = []
    for group in groups:
        if len(group["candidate_ids"]) < min_group_size:
            continue
        y_true = np.asarray([group["relevance"]], dtype=np.float32)
        y_score = np.asarray([[score_map[candidate_id] for candidate_id in group["candidate_ids"]]], dtype=np.float32)
        values.append(float(ndcg_score(y_true, y_score, k=min(k, len(group["candidate_ids"])))))
    if not values:
        return None
    return float(np.mean(values))


def _pairwise_accuracy(pair_rows: list[dict[str, str]], split_by_candidate: dict[str, str], score_map: dict[str, float], split: str) -> tuple[float | None, int]:
    total = 0
    correct = 0
    for row in pair_rows:
        left = row["candidate_id_left"]
        right = row["candidate_id_right"]
        if split_by_candidate.get(left) != split or split_by_candidate.get(right) != split:
            continue
        total += 1
        predicted = left if score_map[left] >= score_map[right] else right
        if predicted == row["preferred_candidate_id"]:
            correct += 1
    if total == 0:
        return None, 0
    return correct / total, total


def _pairwise_arrays(
    pair_rows: list[dict[str, str]],
    rows_by_candidate: dict[str, dict[str, Any]],
    feature_names: list[str],
    split: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    sample_weights: list[float] = []
    for row in pair_rows:
        left = row["candidate_id_left"]
        right = row["candidate_id_right"]
        if left not in rows_by_candidate or right not in rows_by_candidate:
            continue
        if rows_by_candidate[left]["split"] != split or rows_by_candidate[right]["split"] != split:
            continue
        left_vec = rows_by_candidate[left]["feature_map"]
        right_vec = rows_by_candidate[right]["feature_map"]
        x_rows.append([float(left_vec.get(name, 0.0)) - float(right_vec.get(name, 0.0)) for name in feature_names])
        y_rows.append(1 if row["preferred_candidate_id"] == left else 0)
        sample_weights.append(float(row.get("preference_strength") or 1.0))
    return (
        np.asarray(x_rows, dtype=np.float32),
        np.asarray(y_rows, dtype=np.int32),
        np.asarray(sample_weights, dtype=np.float32),
    )


def _candidate_score_map_from_pairwise_model(
    rows: list[dict[str, Any]],
    feature_names: list[str],
    scaler: StandardScaler,
    model: LogisticRegression,
) -> dict[str, float]:
    x = _to_numpy_matrix(rows, feature_names)
    transformed = scaler.transform(x)
    coefficients = np.asarray(model.coef_[0], dtype=np.float32)
    scores = transformed @ coefficients
    return {
        row["candidate_id"]: float(score)
        for row, score in zip(rows, scores, strict=True)
    }


def _slice_binary_metrics(
    rows: list[dict[str, Any]],
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float,
) -> dict[str, dict[str, float | int | None]]:
    metrics: dict[str, dict[str, float | int | None]] = {}
    slice_names = sorted({name for row in rows for name in row["slices"].keys()})
    for slice_name in slice_names:
        indices = [idx for idx, row in enumerate(rows) if row["slices"].get(slice_name)]
        if not indices:
            continue
        y_true_slice = y_true[indices]
        y_score_slice = y_score[indices]
        if len(y_true_slice) < 3:
            continue
        metrics[slice_name] = {
            "row_count": int(len(indices)),
            "positive_count": int(int(np.sum(y_true_slice))),
            **_binary_metrics(y_true_slice, y_score_slice, threshold),
        }
    return metrics


def _slice_priority_metrics(rows: list[dict[str, Any]], y_true_raw: np.ndarray, y_score_norm: np.ndarray) -> dict[str, dict[str, float | int]]:
    metrics: dict[str, dict[str, float | int]] = {}
    slice_names = sorted({name for row in rows for name in row["slices"].keys()})
    for slice_name in slice_names:
        indices = [idx for idx, row in enumerate(rows) if row["slices"].get(slice_name)]
        if len(indices) < 3:
            continue
        metrics[slice_name] = {
            "row_count": int(len(indices)),
            **_priority_metrics(y_true_raw[indices], y_score_norm[indices]),
        }
    return metrics


def _manual_spot_check_paths() -> list[Path]:
    return [
        ROOT / "data" / "experiments" / "cand1" / "cand1.json",
        ROOT / "data" / "experiments" / "cand2" / "cand2.json",
        ROOT / "data" / "experiments" / "cand3" / "cand3.json",
    ]


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    rows = build_or_load_candidate_feature_cache(repr_config=DEFAULT_REPR_CONFIG, rebuild=False)
    split_rows = _split_rows(rows)
    rows_by_candidate = {row["candidate_id"]: row for row in rows}
    feature_names = feature_names_from_rows(rows)
    batch_tasks = load_batch_tasks()
    pairwise_rows = load_pairwise_rows()

    x_train = _to_numpy_matrix(split_rows["train"], feature_names)
    x_val = _to_numpy_matrix(split_rows["validation"], feature_names)
    x_test = _to_numpy_matrix(split_rows["test"], feature_names)

    priority_y_train_raw = np.asarray([row["labels"]["final_committee_priority"] for row in split_rows["train"]], dtype=np.float32)
    priority_y_val_raw = np.asarray([row["labels"]["final_committee_priority"] for row in split_rows["validation"]], dtype=np.float32)
    priority_y_test_raw = np.asarray([row["labels"]["final_committee_priority"] for row in split_rows["test"]], dtype=np.float32)
    priority_y_train = np.asarray([_priority_target(int(value)) for value in priority_y_train_raw], dtype=np.float32)
    priority_y_val = np.asarray([_priority_target(int(value)) for value in priority_y_val_raw], dtype=np.float32)
    priority_y_test = np.asarray([_priority_target(int(value)) for value in priority_y_test_raw], dtype=np.float32)

    priority_baseline_train = np.asarray([row["baseline_outputs"]["committee_priority_score"] for row in split_rows["train"]], dtype=np.float32)
    priority_baseline_val = np.asarray([row["baseline_outputs"]["committee_priority_score"] for row in split_rows["validation"]], dtype=np.float32)
    priority_baseline_test = np.asarray([row["baseline_outputs"]["committee_priority_score"] for row in split_rows["test"]], dtype=np.float32)

    priority_candidates: list[dict[str, Any]] = []
    for spec in _priority_model_specs():
        model = spec.factory()
        model.fit(x_train, priority_y_train)
        train_score = np.clip(np.asarray(model.predict(x_train), dtype=np.float32), 0.0, 1.0)
        val_score = np.clip(np.asarray(model.predict(x_val), dtype=np.float32), 0.0, 1.0)
        test_score = np.clip(np.asarray(model.predict(x_test), dtype=np.float32), 0.0, 1.0)
        priority_candidates.append(
            {
                "model_name": spec.name,
                "model": model,
                "train": _priority_metrics(priority_y_train_raw, train_score),
                "validation": _priority_metrics(priority_y_val_raw, val_score),
                "test": _priority_metrics(priority_y_test_raw, test_score),
                "scores": {
                    "train": train_score,
                    "validation": val_score,
                    "test": test_score,
                },
            }
        )

    priority_best = max(
        priority_candidates,
        key=lambda item: (item["validation"]["spearman"], -item["validation"]["mae"]),
    )
    joblib.dump(
        {
            "model_name": priority_best["model_name"],
            "feature_names": feature_names,
            "model": priority_best["model"],
        },
        PRIORITY_MODEL_JOBLIB,
    )

    routing_targets = {
        "shortlist_band": ("final_shortlist_band", "shortlist_band_score"),
        "hidden_potential_band": ("final_hidden_potential_band", "hidden_potential_band_score"),
        "support_needed_band": ("final_support_needed_band", "support_needed_band_score"),
        "authenticity_review_band": ("final_authenticity_review_band", "authenticity_review_band_score"),
    }
    routing_results: dict[str, Any] = {}
    routing_models_to_save: dict[str, Any] = {}
    slice_summary: dict[str, Any] = {
        "priority_model_v1": {
            "baseline": {
                "validation": _slice_priority_metrics(split_rows["validation"], priority_y_val_raw, priority_baseline_val),
                "test": _slice_priority_metrics(split_rows["test"], priority_y_test_raw, priority_baseline_test),
            },
            "selected_model": {
                "validation": _slice_priority_metrics(split_rows["validation"], priority_y_val_raw, priority_best["scores"]["validation"]),
                "test": _slice_priority_metrics(split_rows["test"], priority_y_test_raw, priority_best["scores"]["test"]),
            },
        }
    }

    for target_name, (label_key, baseline_key) in routing_targets.items():
        y_train = np.asarray([int(row["labels"][label_key]) for row in split_rows["train"]], dtype=np.int32)
        y_val = np.asarray([int(row["labels"][label_key]) for row in split_rows["validation"]], dtype=np.int32)
        y_test = np.asarray([int(row["labels"][label_key]) for row in split_rows["test"]], dtype=np.int32)
        baseline_train = np.asarray([row["baseline_outputs"][baseline_key] for row in split_rows["train"]], dtype=np.float32)
        baseline_val = np.asarray([row["baseline_outputs"][baseline_key] for row in split_rows["validation"]], dtype=np.float32)
        baseline_test = np.asarray([row["baseline_outputs"][baseline_key] for row in split_rows["test"]], dtype=np.float32)

        candidates: list[dict[str, Any]] = []
        for spec in _routing_model_specs():
            model = spec.factory()
            model.fit(x_train, y_train)
            train_score = _predict_scores(model, x_train)
            val_score = _predict_scores(model, x_val)
            test_score = _predict_scores(model, x_test)
            threshold = _best_threshold_for_f1(y_val, val_score)
            candidates.append(
                {
                    "model_name": spec.name,
                    "model": model,
                    "threshold": threshold,
                    "default": {
                        "train": _binary_metrics(y_train, train_score, 0.5),
                        "validation": _binary_metrics(y_val, val_score, 0.5),
                        "test": _binary_metrics(y_test, test_score, 0.5),
                    },
                    "tuned": {
                        "train": _binary_metrics(y_train, train_score, threshold),
                        "validation": _binary_metrics(y_val, val_score, threshold),
                        "test": _binary_metrics(y_test, test_score, threshold),
                    },
                    "scores": {
                        "train": train_score,
                        "validation": val_score,
                        "test": test_score,
                    },
                }
            )

        best = max(
            candidates,
            key=lambda item: (
                item["tuned"]["validation"]["f1"],
                item["tuned"]["validation"]["average_precision"] or -1.0,
            ),
        )
        routing_results[target_name] = {
            "label_key": label_key,
            "baseline": {
                "train": _binary_metrics(y_train, baseline_train, 0.5),
                "validation": _binary_metrics(y_val, baseline_val, 0.5),
                "test": _binary_metrics(y_test, baseline_test, 0.5),
            },
            "selected_model_name": best["model_name"],
            "selected_threshold": best["threshold"],
            "selected_default": best["default"],
            "selected_tuned": best["tuned"],
            "candidates": [
                {
                    "model_name": item["model_name"],
                    "validation_f1_tuned": item["tuned"]["validation"]["f1"],
                    "validation_average_precision": item["tuned"]["validation"]["average_precision"],
                    "test_f1_tuned": item["tuned"]["test"]["f1"],
                    "test_average_precision": item["tuned"]["test"]["average_precision"],
                }
                for item in candidates
            ],
        }
        routing_models_to_save[target_name] = {
            "model_name": best["model_name"],
            "feature_names": feature_names,
            "threshold": best["threshold"],
            "model": best["model"],
        }
        slice_summary[target_name] = {
            "baseline": {
                "validation": _slice_binary_metrics(split_rows["validation"], y_val, baseline_val, 0.5),
                "test": _slice_binary_metrics(split_rows["test"], y_test, baseline_test, 0.5),
            },
            "selected_model": {
                "validation": _slice_binary_metrics(split_rows["validation"], y_val, best["scores"]["validation"], best["threshold"]),
                "test": _slice_binary_metrics(split_rows["test"], y_test, best["scores"]["test"], best["threshold"]),
            },
        }

    joblib.dump(routing_models_to_save, ROUTING_MODEL_JOBLIB)

    pairwise_specs = _pairwise_model_specs()
    pair_train_x, pair_train_y, pair_train_w = _pairwise_arrays(pairwise_rows, rows_by_candidate, feature_names, "train")
    pair_val_x, pair_val_y, _pair_val_w = _pairwise_arrays(pairwise_rows, rows_by_candidate, feature_names, "validation")
    pair_test_x, pair_test_y, _pair_test_w = _pairwise_arrays(pairwise_rows, rows_by_candidate, feature_names, "test")

    pairwise_candidates: list[dict[str, Any]] = []
    for spec in pairwise_specs:
        scaler = StandardScaler(with_mean=False)
        x_train_scaled = scaler.fit_transform(pair_train_x)
        model = LogisticRegression(
            max_iter=2000,
            fit_intercept=False,
            solver="liblinear",
            C=spec.c_value,
            random_state=RANDOM_SEED,
        )
        model.fit(x_train_scaled, pair_train_y, sample_weight=pair_train_w)

        train_prob = model.predict_proba(x_train_scaled)[:, 1]
        val_prob = model.predict_proba(scaler.transform(pair_val_x))[:, 1]
        test_prob = model.predict_proba(scaler.transform(pair_test_x))[:, 1]
        pairwise_candidates.append(
            {
                "model_name": spec.name,
                "scaler": scaler,
                "model": model,
                "train_pairwise_accuracy": float(np.mean((train_prob >= 0.5) == pair_train_y)) if len(pair_train_y) else 0.0,
                "validation_pairwise_accuracy": float(np.mean((val_prob >= 0.5) == pair_val_y)) if len(pair_val_y) else 0.0,
                "test_pairwise_accuracy": float(np.mean((test_prob >= 0.5) == pair_test_y)) if len(pair_test_y) else 0.0,
            }
        )

    pairwise_best = max(
        pairwise_candidates,
        key=lambda item: (item["validation_pairwise_accuracy"], item["test_pairwise_accuracy"]),
    )
    joblib.dump(
        {
            "model_name": pairwise_best["model_name"],
            "feature_names": feature_names,
            "scaler": pairwise_best["scaler"],
            "model": pairwise_best["model"],
        },
        PAIRWISE_MODEL_JOBLIB,
    )

    split_by_candidate = {row["candidate_id"]: row["split"] for row in rows}
    candidate_score_map = _candidate_score_map_from_pairwise_model(
        rows,
        feature_names,
        pairwise_best["scaler"],
        pairwise_best["model"],
    )
    baseline_ranker_score_map = {
        row["candidate_id"]: float(row["baseline_outputs"]["pairwise_ranker_score"])
        for row in rows
    }
    groups_min3 = _build_rank_groups(rows, batch_tasks, min_group_size=3)
    groups_min5 = _build_rank_groups(rows, batch_tasks, min_group_size=5)
    validation_groups3 = [group for group in groups_min3 if group["split"] == "validation"]
    test_groups3 = [group for group in groups_min3 if group["split"] == "test"]
    validation_groups5 = [group for group in groups_min5 if group["split"] == "validation"]
    test_groups5 = [group for group in groups_min5 if group["split"] == "test"]

    baseline_pairwise_val, baseline_pairwise_val_count = _pairwise_accuracy(pairwise_rows, split_by_candidate, baseline_ranker_score_map, "validation")
    baseline_pairwise_test, baseline_pairwise_test_count = _pairwise_accuracy(pairwise_rows, split_by_candidate, baseline_ranker_score_map, "test")
    pairwise_val, pairwise_val_count = _pairwise_accuracy(pairwise_rows, split_by_candidate, candidate_score_map, "validation")
    pairwise_test, pairwise_test_count = _pairwise_accuracy(pairwise_rows, split_by_candidate, candidate_score_map, "test")

    pairwise_result = {
        "selected_model_name": pairwise_best["model_name"],
        "baseline": {
            "validation_pairwise_accuracy": baseline_pairwise_val,
            "validation_pairwise_count": baseline_pairwise_val_count,
            "test_pairwise_accuracy": baseline_pairwise_test,
            "test_pairwise_count": baseline_pairwise_test_count,
            "validation_ndcg_at_3": _mean_ndcg(validation_groups3, baseline_ranker_score_map, 3, 3),
            "validation_ndcg_at_5": _mean_ndcg(validation_groups5, baseline_ranker_score_map, 5, 5),
            "test_ndcg_at_3": _mean_ndcg(test_groups3, baseline_ranker_score_map, 3, 3),
            "test_ndcg_at_5": _mean_ndcg(test_groups5, baseline_ranker_score_map, 5, 5),
        },
        "selected_model": {
            "validation_pairwise_accuracy": pairwise_val,
            "validation_pairwise_count": pairwise_val_count,
            "test_pairwise_accuracy": pairwise_test,
            "test_pairwise_count": pairwise_test_count,
            "validation_ndcg_at_3": _mean_ndcg(validation_groups3, candidate_score_map, 3, 3),
            "validation_ndcg_at_5": _mean_ndcg(validation_groups5, candidate_score_map, 5, 5),
            "test_ndcg_at_3": _mean_ndcg(test_groups3, candidate_score_map, 3, 3),
            "test_ndcg_at_5": _mean_ndcg(test_groups5, candidate_score_map, 5, 5),
        },
        "candidates": [
            {
                "model_name": item["model_name"],
                "validation_pairwise_accuracy": item["validation_pairwise_accuracy"],
                "test_pairwise_accuracy": item["test_pairwise_accuracy"],
            }
            for item in pairwise_candidates
        ],
    }

    metrics_summary = {
        "row_count": len(rows),
        "feature_count": len(feature_names),
        "representation_config": {
            "backend": DEFAULT_REPR_CONFIG.backend,
            "model_name": DEFAULT_REPR_CONFIG.model_name,
            "device": DEFAULT_REPR_CONFIG.device,
        },
        "priority_model_v1": {
            "baseline": {
                "train": _priority_metrics(priority_y_train_raw, priority_baseline_train),
                "validation": _priority_metrics(priority_y_val_raw, priority_baseline_val),
                "test": _priority_metrics(priority_y_test_raw, priority_baseline_test),
            },
            "selected_model_name": priority_best["model_name"],
            "selected_model": {
                "train": priority_best["train"],
                "validation": priority_best["validation"],
                "test": priority_best["test"],
            },
            "candidates": [
                {
                    "model_name": item["model_name"],
                    "validation_spearman": item["validation"]["spearman"],
                    "validation_mae": item["validation"]["mae"],
                    "test_spearman": item["test"]["spearman"],
                    "test_mae": item["test"]["mae"],
                }
                for item in priority_candidates
            ],
        },
        "routing_models_v1": routing_results,
        "pairwise_ranker_v2": pairwise_result,
    }

    with METRICS_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metrics_summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with SLICE_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(slice_summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    priority_scores_all = {
        row["candidate_id"]: float(_predict_scores(priority_best["model"], _to_numpy_matrix([row], feature_names))[0])
        for row in rows
    }
    routing_scores_all: dict[str, dict[str, float]] = {}
    for target_name, bundle in routing_models_to_save.items():
        routing_scores_all[target_name] = {
            row["candidate_id"]: float(_predict_scores(bundle["model"], _to_numpy_matrix([row], feature_names))[0])
            for row in rows
        }

    with PREDICTIONS_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "split",
                "source_group",
                "origin_language_slice",
                "final_committee_priority",
                "final_shortlist_band",
                "final_hidden_potential_band",
                "final_support_needed_band",
                "final_authenticity_review_band",
                "baseline_committee_priority_score",
                "priority_model_score",
                "baseline_shortlist_band_score",
                "shortlist_model_score",
                "baseline_hidden_potential_band_score",
                "hidden_potential_model_score",
                "baseline_support_needed_band_score",
                "support_needed_model_score",
                "baseline_authenticity_review_band_score",
                "authenticity_review_model_score",
                "baseline_pairwise_ranker_score",
                "pairwise_ranker_v2_score",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "candidate_id": row["candidate_id"],
                    "split": row["split"],
                    "source_group": row["source_group"],
                    "origin_language_slice": row["origin_language_slice"],
                    "final_committee_priority": row["labels"]["final_committee_priority"],
                    "final_shortlist_band": str(row["labels"]["final_shortlist_band"]).lower(),
                    "final_hidden_potential_band": str(row["labels"]["final_hidden_potential_band"]).lower(),
                    "final_support_needed_band": str(row["labels"]["final_support_needed_band"]).lower(),
                    "final_authenticity_review_band": str(row["labels"]["final_authenticity_review_band"]).lower(),
                    "baseline_committee_priority_score": row["baseline_outputs"]["committee_priority_score"],
                    "priority_model_score": priority_scores_all[row["candidate_id"]],
                    "baseline_shortlist_band_score": row["baseline_outputs"]["shortlist_band_score"],
                    "shortlist_model_score": routing_scores_all["shortlist_band"][row["candidate_id"]],
                    "baseline_hidden_potential_band_score": row["baseline_outputs"]["hidden_potential_band_score"],
                    "hidden_potential_model_score": routing_scores_all["hidden_potential_band"][row["candidate_id"]],
                    "baseline_support_needed_band_score": row["baseline_outputs"]["support_needed_band_score"],
                    "support_needed_model_score": routing_scores_all["support_needed_band"][row["candidate_id"]],
                    "baseline_authenticity_review_band_score": row["baseline_outputs"]["authenticity_review_band_score"],
                    "authenticity_review_model_score": routing_scores_all["authenticity_review_band"][row["candidate_id"]],
                    "baseline_pairwise_ranker_score": row["baseline_outputs"]["pairwise_ranker_score"],
                    "pairwise_ranker_v2_score": candidate_score_map[row["candidate_id"]],
                }
            )

    manual_spot_checks: list[dict[str, Any]] = []
    for path in _manual_spot_check_paths():
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        external_row = build_feature_row_for_payload(payload, repr_config=DEFAULT_REPR_CONFIG)
        x_external = np.asarray(
            [[float(external_row["feature_map"].get(name, 0.0)) for name in feature_names]],
            dtype=np.float32,
        )
        pairwise_external_score = _candidate_score_map_from_pairwise_model(
            [{"candidate_id": external_row["candidate_id"], "feature_map": external_row["feature_map"]}],
            feature_names,
            pairwise_best["scaler"],
            pairwise_best["model"],
        )[external_row["candidate_id"]]
        manual_spot_checks.append(
            {
                "path": str(path),
                "candidate_id": external_row["candidate_id"],
                "baseline_outputs": external_row["baseline_outputs"],
                "priority_model_score": float(_predict_scores(priority_best["model"], x_external)[0]),
                "routing_model_scores": {
                    target_name: float(_predict_scores(bundle["model"], x_external)[0])
                    for target_name, bundle in routing_models_to_save.items()
                },
                "pairwise_ranker_v2_score": float(pairwise_external_score),
                "raw_recommendation": external_row["raw_recommendation"],
            }
        )

    with SPOT_CHECKS_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manual_spot_checks, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    report_lines = [
        "# Offline ML/NLP Layer V1",
        "",
        f"- row_count: `{len(rows)}`",
        f"- feature_count: `{len(feature_names)}`",
        f"- representation backend: `{DEFAULT_REPR_CONFIG.backend}`",
        f"- representation model: `{DEFAULT_REPR_CONFIG.model_name}`",
        "",
        "## Priority Model",
        "",
        f"- baseline validation spearman: `{metrics_summary['priority_model_v1']['baseline']['validation']['spearman']:.4f}`",
        f"- selected validation spearman: `{metrics_summary['priority_model_v1']['selected_model']['validation']['spearman']:.4f}`",
        f"- baseline test spearman: `{metrics_summary['priority_model_v1']['baseline']['test']['spearman']:.4f}`",
        f"- selected test spearman: `{metrics_summary['priority_model_v1']['selected_model']['test']['spearman']:.4f}`",
        "",
        "## Routing Models",
        "",
    ]
    for target_name, payload in routing_results.items():
        report_lines.extend(
            [
                f"- `{target_name}` baseline val F1: `{payload['baseline']['validation']['f1']:.4f}` -> selected val F1: `{payload['selected_tuned']['validation']['f1']:.4f}`",
                f"- `{target_name}` baseline test F1: `{payload['baseline']['test']['f1']:.4f}` -> selected test F1: `{payload['selected_tuned']['test']['f1']:.4f}`",
            ]
        )
    report_lines.extend(
        [
            "",
            "## Pairwise Ranker",
            "",
            f"- baseline validation pairwise accuracy: `{pairwise_result['baseline']['validation_pairwise_accuracy']}`",
            f"- selected validation pairwise accuracy: `{pairwise_result['selected_model']['validation_pairwise_accuracy']}`",
            f"- baseline test pairwise accuracy: `{pairwise_result['baseline']['test_pairwise_accuracy']}`",
            f"- selected test pairwise accuracy: `{pairwise_result['selected_model']['test_pairwise_accuracy']}`",
            "",
            "## Notes",
            "",
            "- This layer is offline-only and does not change runtime scoring yet.",
            "- Promotion into runtime should only happen if the selected model beats baseline on validation and test without slice regressions beyond the agreed threshold.",
            "- Manual spot checks for `cand1/cand2/cand3` are exported when those payload files exist locally.",
        ]
    )
    REPORT_MD.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(metrics_summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
