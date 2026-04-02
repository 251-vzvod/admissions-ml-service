from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from app.services.pipeline import ScoringPipeline
from research.scripts.train_shortlist_ranker_v1 import EXPORTS_DIR, TRAINING_CSV, build_feature_matrix, load_payloads


MODEL_DIR = EXPORTS_DIR / "models" / "manual_review_probe_v1_training_dataset_v3"
METRICS_JSON = MODEL_DIR / "metrics_summary.json"
COEFFICIENTS_CSV = MODEL_DIR / "coefficients.csv"
PREDICTIONS_CSV = MODEL_DIR / "candidate_predictions.csv"
REPORT_MD = MODEL_DIR / "probe_report.md"


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def load_training_rows() -> list[dict[str, Any]]:
    with TRAINING_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def binary_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float) -> dict[str, float | None]:
    if len(np.unique(y_true)) < 2:
        auc = None
        ap = None
    else:
        auc = float(roc_auc_score(y_true, y_score))
        ap = float(average_precision_score(y_true, y_score))
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
    for threshold in np.linspace(0.1, 0.9, 17):
        y_pred = (y_score >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = float(threshold)
    return best_threshold


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_training_rows()
    payloads = load_payloads()
    pipeline = ScoringPipeline()
    feature_names, feature_map_by_candidate, _baseline_score_map = build_feature_matrix(pipeline, payloads)

    split_to_ids: dict[str, list[str]] = {"train": [], "validation": [], "test": []}
    target_by_candidate: dict[str, int] = {}
    baseline_by_candidate: dict[str, float] = {}
    meta_by_candidate: dict[str, dict[str, Any]] = {}
    split_by_candidate: dict[str, str] = {}

    for row in rows:
        candidate_id = row["candidate_id"]
        split = row["split"]
        split_by_candidate[candidate_id] = split
        split_to_ids[split].append(candidate_id)
        manual_target = 1 if row["final_recommendation"] == "manual_review_required" else 0
        target_by_candidate[candidate_id] = manual_target
        baseline_by_candidate[candidate_id] = 1.0 - float(feature_map_by_candidate[candidate_id]["authenticity_risk_neg"])
        meta_by_candidate[candidate_id] = {
            "source_group": row["source_group"],
            "final_recommendation": row["final_recommendation"],
            "final_authenticity_review_band": parse_bool(row["final_authenticity_review_band"]),
            "final_support_needed_band": parse_bool(row["final_support_needed_band"]),
        }

    def matrix(candidate_ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(
            [[float(feature_map_by_candidate[candidate_id][name]) for name in feature_names] for candidate_id in candidate_ids],
            dtype=np.float32,
        )
        y = np.asarray([target_by_candidate[candidate_id] for candidate_id in candidate_ids], dtype=np.int32)
        return x, y

    train_ids = split_to_ids["train"]
    val_ids = split_to_ids["validation"]
    test_ids = split_to_ids["test"]

    x_train, y_train = matrix(train_ids)
    x_val, y_val = matrix(val_ids)
    x_test, y_test = matrix(test_ids)

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=20260402,
        solver="liblinear",
    )
    model.fit(x_train, y_train)

    val_score = model.predict_proba(x_val)[:, 1]
    test_score = model.predict_proba(x_test)[:, 1]
    train_score = model.predict_proba(x_train)[:, 1]

    baseline_val = np.asarray([baseline_by_candidate[candidate_id] for candidate_id in val_ids], dtype=np.float32)
    baseline_test = np.asarray([baseline_by_candidate[candidate_id] for candidate_id in test_ids], dtype=np.float32)
    baseline_train = np.asarray([baseline_by_candidate[candidate_id] for candidate_id in train_ids], dtype=np.float32)

    tuned_threshold = best_threshold_for_f1(y_val, val_score)

    metrics = {
        "row_count": len(rows),
        "target_name": "final_recommendation == manual_review_required",
        "feature_names": feature_names,
        "class_counts": {
            split: {
                "total": len(candidate_ids),
                "manual_review_positive": int(sum(target_by_candidate[candidate_id] for candidate_id in candidate_ids)),
            }
            for split, candidate_ids in split_to_ids.items()
        },
        "thresholds": {
            "default": 0.5,
            "validation_best_f1": tuned_threshold,
        },
        "baseline_authenticity_risk": {
            "train": binary_metrics(y_train, baseline_train, 0.5),
            "validation": binary_metrics(y_val, baseline_val, 0.5),
            "test": binary_metrics(y_test, baseline_test, 0.5),
        },
        "logistic_probe_default_threshold": {
            "train": binary_metrics(y_train, train_score, 0.5),
            "validation": binary_metrics(y_val, val_score, 0.5),
            "test": binary_metrics(y_test, test_score, 0.5),
        },
        "logistic_probe_tuned_threshold": {
            "train": binary_metrics(y_train, train_score, tuned_threshold),
            "validation": binary_metrics(y_val, val_score, tuned_threshold),
            "test": binary_metrics(y_test, test_score, tuned_threshold),
        },
    }

    with METRICS_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    coefficients = sorted(
        (
            {"feature_name": name, "coefficient": float(coef)}
            for name, coef in zip(feature_names, model.coef_[0], strict=True)
        ),
        key=lambda item: abs(item["coefficient"]),
        reverse=True,
    )
    with COEFFICIENTS_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["feature_name", "coefficient"])
        writer.writeheader()
        writer.writerows(coefficients)

    with PREDICTIONS_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "candidate_id",
                "split",
                "target_manual_review",
                "baseline_authenticity_risk_score",
                "probe_probability",
                "source_group",
                "final_recommendation",
                "final_authenticity_review_band",
                "final_support_needed_band",
            ],
        )
        writer.writeheader()
        for candidate_id in [row["candidate_id"] for row in rows]:
            split = split_by_candidate[candidate_id]
            if split == "train":
                score = train_score[train_ids.index(candidate_id)]
                baseline = baseline_train[train_ids.index(candidate_id)]
            elif split == "validation":
                score = val_score[val_ids.index(candidate_id)]
                baseline = baseline_val[val_ids.index(candidate_id)]
            else:
                score = test_score[test_ids.index(candidate_id)]
                baseline = baseline_test[test_ids.index(candidate_id)]
            writer.writerow(
                {
                    "candidate_id": candidate_id,
                    "split": split,
                    "target_manual_review": target_by_candidate[candidate_id],
                    "baseline_authenticity_risk_score": float(baseline),
                    "probe_probability": float(score),
                    "source_group": meta_by_candidate[candidate_id]["source_group"],
                    "final_recommendation": meta_by_candidate[candidate_id]["final_recommendation"],
                    "final_authenticity_review_band": str(meta_by_candidate[candidate_id]["final_authenticity_review_band"]).lower(),
                    "final_support_needed_band": str(meta_by_candidate[candidate_id]["final_support_needed_band"]).lower(),
                }
            )

    top_pos = coefficients[:6]
    report_lines = [
        "# Manual Review Probe V1",
        "",
        "- target: `final_recommendation == manual_review_required`",
        f"- validation best-F1 threshold: `{tuned_threshold:.2f}`",
        "",
        "## Class Counts",
        "",
    ]
    for split, counts in metrics["class_counts"].items():
        report_lines.append(
            f"- {split}: total `{counts['total']}`, manual_review_positive `{counts['manual_review_positive']}`"
        )
    report_lines.extend(
        [
            "",
            "## Validation/Test Snapshot",
            "",
            f"- baseline validation AP: `{metrics['baseline_authenticity_risk']['validation']['average_precision']}`",
            f"- probe validation AP: `{metrics['logistic_probe_default_threshold']['validation']['average_precision']}`",
            f"- baseline test AP: `{metrics['baseline_authenticity_risk']['test']['average_precision']}`",
            f"- probe test AP: `{metrics['logistic_probe_default_threshold']['test']['average_precision']}`",
            "",
            "## Strongest Coefficients",
            "",
        ]
    )
    for row in top_pos:
        report_lines.append(f"- `{row['feature_name']}`: `{row['coefficient']:.4f}`")
    REPORT_MD.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
