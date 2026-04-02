from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from app.services.pipeline import ScoringPipeline
from research.scripts.train_shortlist_ranker_v1 import EXPORTS_DIR, TRAINING_CSV, build_feature_matrix, load_payloads


MODEL_DIR = EXPORTS_DIR / "models" / "manual_review_probe_v2_training_dataset_v3"
METRICS_JSON = MODEL_DIR / "metrics_summary.json"
EXPERIMENTS_CSV = MODEL_DIR / "experiments.csv"
PREDICTIONS_CSV = MODEL_DIR / "candidate_predictions.csv"
FEATURES_CSV = MODEL_DIR / "selected_model_features.csv"
REPORT_MD = MODEL_DIR / "probe_report.md"
MODEL_JOBLIB = MODEL_DIR / "selected_model.joblib"
RUNTIME_ASSET_DIR = ROOT / "app" / "assets"
RUNTIME_ARTIFACT_NAME = "review_routing_sidecar_v1"
RUNTIME_MODEL_JOBLIB = RUNTIME_ASSET_DIR / f"{RUNTIME_ARTIFACT_NAME}.joblib"
RUNTIME_METADATA_JSON = RUNTIME_ASSET_DIR / f"{RUNTIME_ARTIFACT_NAME}.json"


@dataclass(frozen=True)
class TargetSpec:
    name: str
    description: str
    fn: Callable[[dict[str, str]], bool]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    factory: Callable[[], Any]


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def load_training_rows() -> list[dict[str, str]]:
    with TRAINING_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | None]:
    auc = float(roc_auc_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None
    ap = float(average_precision_score(y_true, y_score)) if len(np.unique(y_true)) > 1 else None
    y_pred = (y_score >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "roc_auc": auc,
        "average_precision": ap,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }


def best_threshold_for_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.15, 0.85, 15):
        y_pred = (y_score >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def build_model_specs() -> list[ModelSpec]:
    return [
        ModelSpec(
            name="logreg_balanced",
            factory=lambda: LogisticRegression(
                max_iter=2000,
                class_weight="balanced",
                random_state=20260402,
                solver="liblinear",
            ),
        ),
        ModelSpec(
            name="random_forest_balanced",
            factory=lambda: RandomForestClassifier(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=2,
                random_state=20260402,
                class_weight="balanced_subsample",
                n_jobs=1,
            ),
        ),
    ]


def build_target_specs() -> list[TargetSpec]:
    return [
        TargetSpec(
            name="manual_review_required",
            description="Strict manual review routing target.",
            fn=lambda row: row["final_recommendation"] == "manual_review_required",
        ),
        TargetSpec(
            name="nonstandard_route",
            description="Non-standard routing: manual review or insufficient evidence.",
            fn=lambda row: row["final_recommendation"] in {"manual_review_required", "insufficient_evidence"},
        ),
        TargetSpec(
            name="review_risk_or_insufficient",
            description="Review-risk or insufficient-evidence routing target.",
            fn=lambda row: (
                row["final_recommendation"] == "manual_review_required"
                or parse_bool(row["final_authenticity_review_band"])
                or row["final_recommendation"] == "insufficient_evidence"
            ),
        ),
    ]


def baseline_score_for_row(
    candidate_id: str,
    feature_map_by_candidate: dict[str, dict[str, float]],
    baseline_name: str,
) -> float:
    feature_map = feature_map_by_candidate[candidate_id]
    authenticity_risk_score = 1.0 - float(feature_map["authenticity_risk_neg"])
    confidence_score = 1.0 - float(feature_map["confidence_score"])
    if baseline_name == "authenticity_risk_only":
        return authenticity_risk_score
    if baseline_name == "authenticity_plus_low_confidence":
        return (authenticity_risk_score * 0.7) + (confidence_score * 0.3)
    raise ValueError(f"Unsupported baseline: {baseline_name}")


def feature_importance_rows(model: Any, feature_names: list[str]) -> list[dict[str, float | str]]:
    if hasattr(model, "coef_"):
        values = [float(value) for value in model.coef_[0]]
    elif hasattr(model, "feature_importances_"):
        values = [float(value) for value in model.feature_importances_]
    else:
        values = [0.0 for _ in feature_names]
    rows = [
        {
            "feature_name": name,
            "importance": value,
        }
        for name, value in zip(feature_names, values, strict=True)
    ]
    rows.sort(key=lambda item: abs(float(item["importance"])), reverse=True)
    return rows


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_training_rows()
    payloads = load_payloads()
    pipeline = ScoringPipeline()
    feature_names, feature_map_by_candidate, _ = build_feature_matrix(pipeline, payloads)

    split_to_ids: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    split_by_candidate: dict[str, str] = {}
    meta_by_candidate: dict[str, dict[str, Any]] = {}
    for row in rows:
        candidate_id = row["candidate_id"]
        split = row["split"]
        split_to_ids[split].append(candidate_id)
        split_by_candidate[candidate_id] = split
        meta_by_candidate[candidate_id] = {
            "source_group": row["source_group"],
            "final_recommendation": row["final_recommendation"],
            "final_authenticity_review_band": parse_bool(row["final_authenticity_review_band"]),
            "final_support_needed_band": parse_bool(row["final_support_needed_band"]),
        }

    target_specs = build_target_specs()
    model_specs = build_model_specs()
    baseline_names = ["authenticity_risk_only", "authenticity_plus_low_confidence"]

    experiment_rows: list[dict[str, Any]] = []
    best_result: dict[str, Any] | None = None

    for target_spec in target_specs:
        target_by_candidate = {
            row["candidate_id"]: int(target_spec.fn(row))
            for row in rows
        }

        class_counts = {
            split: {
                "total": len(candidate_ids),
                "positive": int(sum(target_by_candidate[candidate_id] for candidate_id in candidate_ids)),
            }
            for split, candidate_ids in split_to_ids.items()
        }

        matrices: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for split, candidate_ids in split_to_ids.items():
            x = np.asarray(
                [[float(feature_map_by_candidate[candidate_id][name]) for name in feature_names] for candidate_id in candidate_ids],
                dtype=np.float32,
            )
            y = np.asarray([target_by_candidate[candidate_id] for candidate_id in candidate_ids], dtype=np.int32)
            matrices[split] = (x, y)

        x_train, y_train = matrices["train"]
        x_val, y_val = matrices["validation"]
        x_test, y_test = matrices["test"]

        baseline_metrics: dict[str, Any] = {}
        for baseline_name in baseline_names:
            baseline_train = np.asarray(
                [baseline_score_for_row(candidate_id, feature_map_by_candidate, baseline_name) for candidate_id in split_to_ids["train"]],
                dtype=np.float32,
            )
            baseline_val = np.asarray(
                [baseline_score_for_row(candidate_id, feature_map_by_candidate, baseline_name) for candidate_id in split_to_ids["validation"]],
                dtype=np.float32,
            )
            baseline_test = np.asarray(
                [baseline_score_for_row(candidate_id, feature_map_by_candidate, baseline_name) for candidate_id in split_to_ids["test"]],
                dtype=np.float32,
            )
            baseline_metrics[baseline_name] = {
                "train": binary_metrics(y_train, baseline_train, 0.5),
                "validation": binary_metrics(y_val, baseline_val, 0.5),
                "test": binary_metrics(y_test, baseline_test, 0.5),
            }

        for model_spec in model_specs:
            model = model_spec.factory()
            model.fit(x_train, y_train)

            train_score = model.predict_proba(x_train)[:, 1]
            val_score = model.predict_proba(x_val)[:, 1]
            test_score = model.predict_proba(x_test)[:, 1]
            tuned_threshold = best_threshold_for_f1(y_val, val_score)

            result = {
                "target_name": target_spec.name,
                "target_description": target_spec.description,
                "model_name": model_spec.name,
                "class_counts": class_counts,
                "threshold_default": 0.5,
                "threshold_validation_best_f1": tuned_threshold,
                "baseline_metrics": baseline_metrics,
                "default_threshold_metrics": {
                    "train": binary_metrics(y_train, train_score, 0.5),
                    "validation": binary_metrics(y_val, val_score, 0.5),
                    "test": binary_metrics(y_test, test_score, 0.5),
                },
                "tuned_threshold_metrics": {
                    "train": binary_metrics(y_train, train_score, tuned_threshold),
                    "validation": binary_metrics(y_val, val_score, tuned_threshold),
                    "test": binary_metrics(y_test, test_score, tuned_threshold),
                },
                "selected_model": model,
                "feature_names": feature_names,
                "target_by_candidate": target_by_candidate,
                "scores": {
                    "train": train_score,
                    "validation": val_score,
                    "test": test_score,
                },
            }
            experiment_rows.append(result)

            candidate = {
                "validation_average_precision": result["default_threshold_metrics"]["validation"]["average_precision"] or -1.0,
                "validation_roc_auc": result["default_threshold_metrics"]["validation"]["roc_auc"] or -1.0,
                "test_average_precision": result["default_threshold_metrics"]["test"]["average_precision"] or -1.0,
                "result": result,
            }
            if best_result is None:
                best_result = candidate
            else:
                current_key = (
                    candidate["validation_average_precision"],
                    candidate["validation_roc_auc"],
                    candidate["test_average_precision"],
                )
                best_key = (
                    best_result["validation_average_precision"],
                    best_result["validation_roc_auc"],
                    best_result["test_average_precision"],
                )
                if current_key > best_key:
                    best_result = candidate

    if best_result is None:
        raise RuntimeError("No probe experiments were produced.")

    selected = best_result["result"]
    selected_model = selected["selected_model"]
    selected_target_by_candidate = selected["target_by_candidate"]

    metrics_payload = {
        "row_count": len(rows),
        "feature_names": feature_names,
        "selected_target_name": selected["target_name"],
        "selected_target_description": selected["target_description"],
        "selected_model_name": selected["model_name"],
        "thresholds": {
            "default": selected["threshold_default"],
            "validation_best_f1": selected["threshold_validation_best_f1"],
        },
        "selected_default_threshold_metrics": selected["default_threshold_metrics"],
        "selected_tuned_threshold_metrics": selected["tuned_threshold_metrics"],
        "selected_baseline_metrics": selected["baseline_metrics"],
        "all_experiments": [
            {
                "target_name": item["target_name"],
                "model_name": item["model_name"],
                "validation_average_precision": item["default_threshold_metrics"]["validation"]["average_precision"],
                "validation_roc_auc": item["default_threshold_metrics"]["validation"]["roc_auc"],
                "test_average_precision": item["default_threshold_metrics"]["test"]["average_precision"],
                "test_roc_auc": item["default_threshold_metrics"]["test"]["roc_auc"],
                "validation_f1_tuned": item["tuned_threshold_metrics"]["validation"]["f1"],
                "test_f1_tuned": item["tuned_threshold_metrics"]["test"]["f1"],
            }
            for item in experiment_rows
        ],
    }

    with METRICS_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with EXPERIMENTS_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "target_name",
                "model_name",
                "validation_average_precision",
                "validation_roc_auc",
                "test_average_precision",
                "test_roc_auc",
                "validation_f1_tuned",
                "test_f1_tuned",
            ],
        )
        writer.writeheader()
        for row in metrics_payload["all_experiments"]:
            writer.writerow(row)

    feature_rows = feature_importance_rows(selected_model, feature_names)
    with FEATURES_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature_name", "importance"])
        writer.writeheader()
        writer.writerows(feature_rows)

    joblib.dump(selected_model, MODEL_JOBLIB)
    RUNTIME_ASSET_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(selected_model, RUNTIME_MODEL_JOBLIB)
    runtime_metadata = {
        "artifact_name": RUNTIME_ARTIFACT_NAME,
        "artifact_version": "review-routing-sidecar-v1",
        "target_name": selected["target_name"],
        "target_description": selected["target_description"],
        "model_name": selected["model_name"],
        "threshold": selected["threshold_validation_best_f1"],
        "feature_names": feature_names,
        "generated_from": str(MODEL_DIR),
        "note": "Shadow-mode routing sidecar. Does not override deterministic recommendation logic.",
    }
    with RUNTIME_METADATA_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(runtime_metadata, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    with PREDICTIONS_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "split",
                "source_group",
                "target_positive",
                "selected_model_probability",
                "final_recommendation",
                "final_authenticity_review_band",
                "final_support_needed_band",
            ],
        )
        writer.writeheader()
        for candidate_id in [row["candidate_id"] for row in rows]:
            split = split_by_candidate[candidate_id]
            split_ids = split_to_ids[split]
            split_scores = selected["scores"][split]
            score = float(split_scores[split_ids.index(candidate_id)])
            writer.writerow(
                {
                    "candidate_id": candidate_id,
                    "split": split,
                    "source_group": meta_by_candidate[candidate_id]["source_group"],
                    "target_positive": selected_target_by_candidate[candidate_id],
                    "selected_model_probability": score,
                    "final_recommendation": meta_by_candidate[candidate_id]["final_recommendation"],
                    "final_authenticity_review_band": str(meta_by_candidate[candidate_id]["final_authenticity_review_band"]).lower(),
                    "final_support_needed_band": str(meta_by_candidate[candidate_id]["final_support_needed_band"]).lower(),
                }
            )

    top_features = feature_rows[:6]
    report_lines = [
        "# Manual Review Probe V2",
        "",
        f"- selected target: `{selected['target_name']}`",
        f"- selected model: `{selected['model_name']}`",
        f"- validation best-F1 threshold: `{selected['threshold_validation_best_f1']:.2f}`",
        "",
        "## Why V2 Exists",
        "",
        "- V1 treated strict `manual_review_required` as the only target and was too unstable on the frozen GT.",
        "- V2 compares stricter and broader review-routing targets before selecting a sidecar candidate.",
        "- This keeps shortlist ranking separate from review-risk routing.",
        "",
        "## Selected Validation/Test Snapshot",
        "",
        f"- validation AP: `{selected['default_threshold_metrics']['validation']['average_precision']}`",
        f"- validation ROC AUC: `{selected['default_threshold_metrics']['validation']['roc_auc']}`",
        f"- test AP: `{selected['default_threshold_metrics']['test']['average_precision']}`",
        f"- test ROC AUC: `{selected['default_threshold_metrics']['test']['roc_auc']}`",
        "",
        "## Top Selected Model Features",
        "",
    ]
    for row in top_features:
        report_lines.append(f"- `{row['feature_name']}`: `{float(row['importance']):.4f}`")
    report_lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This probe is still offline-only and not a runtime promotion by itself.",
            "- It is intended as a routing sidecar, not as a replacement for deterministic recommendation guardrails.",
            "- `/rank` should remain focused on shortlist ordering; review-risk should be handled by a separate layer.",
        ]
    )
    REPORT_MD.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
