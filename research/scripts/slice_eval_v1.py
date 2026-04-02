from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from sklearn.metrics import ndcg_score


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "ml_workbench"
EXPORTS_DIR = DATA_ROOT / "exports"
LABELS_DIR = DATA_ROOT / "labels"
MODEL_DIR = EXPORTS_DIR / "models" / "shortlist_ranker_v1_training_dataset_v3"

TRAINING_CSV = EXPORTS_DIR / "training_dataset_v3.csv"
PAIRWISE_CSV = LABELS_DIR / "pairwise_labels.csv"
BATCH_JSONL = LABELS_DIR / "batch_shortlist_tasks.jsonl"
PREDICTIONS_CSV = MODEL_DIR / "candidate_predictions.csv"
METRICS_JSON = MODEL_DIR / "metrics_summary.json"

REPORT_MD = MODEL_DIR / "slice_eval_report.md"
REPORT_JSON = MODEL_DIR / "slice_eval_summary.json"


@dataclass(frozen=True)
class CandidateRow:
    candidate_id: str
    split: str
    source_group: str
    origin_language_slice: str
    text_length_bucket: str
    has_interview_text: bool
    recommendation: str
    shortlist_band: bool
    hidden_potential_band: bool
    support_needed_band: bool
    authenticity_review_band: bool


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def load_training_rows() -> dict[str, CandidateRow]:
    rows: dict[str, CandidateRow] = {}
    with TRAINING_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["candidate_id"]] = CandidateRow(
                candidate_id=row["candidate_id"],
                split=row["split"],
                source_group=row["source_group"],
                origin_language_slice=row["origin_language_slice"],
                text_length_bucket=row["text_length_bucket"],
                has_interview_text=parse_bool(row["has_interview_text"]),
                recommendation=row["final_recommendation"],
                shortlist_band=parse_bool(row["final_shortlist_band"]),
                hidden_potential_band=parse_bool(row["final_hidden_potential_band"]),
                support_needed_band=parse_bool(row["final_support_needed_band"]),
                authenticity_review_band=parse_bool(row["final_authenticity_review_band"]),
            )
    return rows


def load_predictions() -> dict[str, dict[str, float]]:
    rows: dict[str, dict[str, float]] = {}
    with PREDICTIONS_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[row["candidate_id"]] = {
                "baseline": float(row["baseline_score"]),
                "learned": float(row["learned_score"]),
            }
    return rows


def load_pairwise_rows() -> list[dict[str, str]]:
    with PAIRWISE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def load_batches() -> list[dict[str, Any]]:
    with BATCH_JSONL.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_metrics() -> dict[str, Any]:
    return json.loads(METRICS_JSON.read_text(encoding="utf-8"))


def pairwise_accuracy(rows: list[dict[str, str]], candidates: dict[str, CandidateRow], predictions: dict[str, dict[str, float]], split: str, model_key: str) -> tuple[float | None, int]:
    total = 0
    correct = 0
    for row in rows:
        left = row["candidate_id_left"]
        right = row["candidate_id_right"]
        if candidates[left].split != split or candidates[right].split != split:
            continue
        total += 1
        predicted = left if predictions[left][model_key] >= predictions[right][model_key] else right
        if predicted == row["preferred_candidate_id"]:
            correct += 1
    if not total:
        return None, 0
    return correct / total, total


def pairwise_accuracy_by_slice(
    rows: list[dict[str, str]],
    candidates: dict[str, CandidateRow],
    predictions: dict[str, dict[str, float]],
    split: str,
    model_key: str,
    slice_name: str,
) -> dict[str, dict[str, float | int]]:
    bucket: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        left = row["candidate_id_left"]
        right = row["candidate_id_right"]
        if candidates[left].split != split or candidates[right].split != split:
            continue

        preferred = row["preferred_candidate_id"]
        preferred_row = candidates[preferred]
        if slice_name == "source_group":
            key = preferred_row.source_group
        elif slice_name == "recommendation":
            key = preferred_row.recommendation
        elif slice_name == "hidden_potential":
            key = str(preferred_row.hidden_potential_band).lower()
        elif slice_name == "support_needed":
            key = str(preferred_row.support_needed_band).lower()
        elif slice_name == "authenticity_review":
            key = str(preferred_row.authenticity_review_band).lower()
        elif slice_name == "has_interview_text":
            key = str(preferred_row.has_interview_text).lower()
        elif slice_name == "text_length_bucket":
            key = preferred_row.text_length_bucket
        elif slice_name == "origin_language_slice":
            key = preferred_row.origin_language_slice
        elif slice_name == "pair_source_relation":
            key = "same_source" if candidates[left].source_group == candidates[right].source_group else "cross_source"
        else:
            raise ValueError(f"Unknown slice: {slice_name}")

        predicted = left if predictions[left][model_key] >= predictions[right][model_key] else right
        bucket[key].append(predicted == preferred)

    result: dict[str, dict[str, float | int]] = {}
    for key, values in bucket.items():
        result[key] = {
            "count": len(values),
            "accuracy": mean(values) if values else 0.0,
        }
    return result


def project_groups(batches: list[dict[str, Any]], candidates: dict[str, CandidateRow], split: str, min_size: int) -> list[list[str]]:
    groups: list[list[str]] = []
    for batch in batches:
        ids = [candidate_id for candidate_id in batch["ranked_candidate_ids"] if candidates[candidate_id].split == split]
        if len(ids) >= min_size:
            groups.append(ids)
    return groups


def mean_ndcg(groups: list[list[str]], predictions: dict[str, dict[str, float]], model_key: str, k: int) -> float | None:
    values: list[float] = []
    for ids in groups:
        size = len(ids)
        y_true = [[float(size - idx) for idx in range(size)]]
        y_score = [[predictions[candidate_id][model_key] for candidate_id in ids]]
        values.append(float(ndcg_score(y_true, y_score, k=min(k, size))))
    if not values:
        return None
    return mean(values)


def shortlist_recall_at_k(groups: list[list[str]], batches: list[dict[str, Any]], candidates: dict[str, CandidateRow], predictions: dict[str, dict[str, float]], split: str, model_key: str, k: int) -> tuple[float | None, int]:
    batch_by_id = {batch["batch_id"]: batch for batch in batches}
    recalls: list[float] = []
    evaluated = 0
    for batch in batches:
        ids = [candidate_id for candidate_id in batch["ranked_candidate_ids"] if candidates[candidate_id].split == split]
        if len(ids) < k:
            continue
        positives = [cid for cid in ids if cid in set(batch["selected_shortlist_candidate_ids"])]
        if not positives:
            continue
        ranked = sorted(ids, key=lambda candidate_id: predictions[candidate_id][model_key], reverse=True)
        topk = set(ranked[:k])
        recalls.append(len(topk.intersection(positives)) / len(positives))
        evaluated += 1
    if not recalls:
        return None, 0
    return mean(recalls), evaluated


def disagreement_examples(rows: list[dict[str, str]], candidates: dict[str, CandidateRow], predictions: dict[str, dict[str, float]], split: str) -> dict[str, list[dict[str, Any]]]:
    learned_wins: list[dict[str, Any]] = []
    baseline_wins: list[dict[str, Any]] = []

    for row in rows:
        left = row["candidate_id_left"]
        right = row["candidate_id_right"]
        if candidates[left].split != split or candidates[right].split != split:
            continue

        preferred = row["preferred_candidate_id"]
        baseline_pred = left if predictions[left]["baseline"] >= predictions[right]["baseline"] else right
        learned_pred = left if predictions[left]["learned"] >= predictions[right]["learned"] else right

        if baseline_pred == learned_pred:
            continue

        item = {
            "pair_id": row["pair_id"],
            "preferred_candidate_id": preferred,
            "left": left,
            "right": right,
            "left_source": candidates[left].source_group,
            "right_source": candidates[right].source_group,
            "preferred_hidden_potential": candidates[preferred].hidden_potential_band,
            "preferred_support_needed": candidates[preferred].support_needed_band,
            "preferred_authenticity_review": candidates[preferred].authenticity_review_band,
            "baseline_pred": baseline_pred,
            "learned_pred": learned_pred,
        }
        if learned_pred == preferred and baseline_pred != preferred:
            learned_wins.append(item)
        elif baseline_pred == preferred and learned_pred != preferred:
            baseline_wins.append(item)

    return {
        "learned_correct_baseline_wrong": learned_wins[:12],
        "baseline_correct_learned_wrong": baseline_wins[:12],
    }


def write_report(summary: dict[str, Any]) -> None:
    with REPORT_MD.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Slice Eval Report\n\n")
        handle.write("## Snapshot\n\n")
        handle.write(f"- dataset: `training_dataset_v3`\n")
        handle.write(f"- row_count: `{summary['row_count']}`\n")
        handle.write(f"- model_dir: `{MODEL_DIR}`\n\n")

        handle.write("## Overall\n\n")
        overall = summary["overall"]
        handle.write(f"- validation NDCG@3 baseline -> learned: `{overall['validation_ndcg_at_3_baseline']:.4f}` -> `{overall['validation_ndcg_at_3_learned']:.4f}`\n")
        handle.write(f"- validation pairwise accuracy baseline -> learned: `{overall['validation_pairwise_accuracy_baseline']:.4f}` -> `{overall['validation_pairwise_accuracy_learned']:.4f}`\n")
        handle.write(f"- test pairwise accuracy baseline -> learned: `{overall['test_pairwise_accuracy_baseline']:.4f}` -> `{overall['test_pairwise_accuracy_learned']:.4f}`\n")
        if overall["test_ndcg_at_3_baseline"] is not None:
            handle.write(f"- test NDCG@3 baseline -> learned: `{overall['test_ndcg_at_3_baseline']:.4f}` -> `{overall['test_ndcg_at_3_learned']:.4f}`\n")
        handle.write("\n")

        handle.write("## Slice Findings\n\n")
        for finding in summary["findings"]:
            handle.write(f"- {finding}\n")

        handle.write("\n## Validation Pairwise By Source\n\n")
        for key, row in sorted(summary["validation_pairwise_by_source"].items()):
            handle.write(f"- {key}: count `{row['count']}`, baseline `{row['baseline_accuracy']:.4f}`, learned `{row['learned_accuracy']:.4f}`\n")

        handle.write("\n## Test Pairwise By Source\n\n")
        for key, row in sorted(summary["test_pairwise_by_source"].items()):
            handle.write(f"- {key}: count `{row['count']}`, baseline `{row['baseline_accuracy']:.4f}`, learned `{row['learned_accuracy']:.4f}`\n")

        handle.write("\n## Validation Pairwise By Slice\n\n")
        for slice_name, slice_rows in summary["validation_pairwise_by_slice"].items():
            handle.write(f"### {slice_name}\n")
            for key, row in sorted(slice_rows.items()):
                handle.write(f"- {key}: count `{row['count']}`, baseline `{row['baseline_accuracy']:.4f}`, learned `{row['learned_accuracy']:.4f}`\n")
            handle.write("\n")

        handle.write("## Test Pairwise By Slice\n\n")
        for slice_name, slice_rows in summary["test_pairwise_by_slice"].items():
            handle.write(f"### {slice_name}\n")
            for key, row in sorted(slice_rows.items()):
                handle.write(f"- {key}: count `{row['count']}`, baseline `{row['baseline_accuracy']:.4f}`, learned `{row['learned_accuracy']:.4f}`\n")
            handle.write("\n")

        handle.write("## Disagreement Examples\n\n")
        handle.write("### Learned Correct / Baseline Wrong\n")
        for item in summary["disagreements"]["learned_correct_baseline_wrong"]:
            handle.write(
                f"- `{item['pair_id']}` preferred `{item['preferred_candidate_id']}` over `{item['left']}` / `{item['right']}` "
                f"(sources: `{item['left_source']}` vs `{item['right_source']}`)\n"
            )
        handle.write("\n### Baseline Correct / Learned Wrong\n")
        for item in summary["disagreements"]["baseline_correct_learned_wrong"]:
            handle.write(
                f"- `{item['pair_id']}` preferred `{item['preferred_candidate_id']}` over `{item['left']}` / `{item['right']}` "
                f"(sources: `{item['left_source']}` vs `{item['right_source']}`)\n"
            )

    with REPORT_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def merged_slice_rows(
    baseline_rows: dict[str, dict[str, float | int]],
    learned_rows: dict[str, dict[str, float | int]],
    min_count: int = 3,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for key in sorted(set(baseline_rows) | set(learned_rows)):
        b = baseline_rows.get(key)
        l = learned_rows.get(key)
        count = int((b or l or {}).get("count", 0))
        if count < min_count:
            continue
        result[key] = {
            "count": count,
            "baseline_accuracy": float((b or {}).get("accuracy", 0.0)),
            "learned_accuracy": float((l or {}).get("accuracy", 0.0)),
        }
    return result


def main() -> None:
    candidates = load_training_rows()
    predictions = load_predictions()
    pairwise_rows = load_pairwise_rows()
    batches = load_batches()
    metrics = load_metrics()

    validation_groups = project_groups(batches, candidates, split="validation", min_size=3)
    test_groups_min2 = project_groups(batches, candidates, split="test", min_size=2)
    test_groups_min3 = project_groups(batches, candidates, split="test", min_size=3)

    slice_names = [
        "source_group",
        "recommendation",
        "hidden_potential",
        "support_needed",
        "authenticity_review",
        "has_interview_text",
        "text_length_bucket",
        "origin_language_slice",
        "pair_source_relation",
    ]

    validation_pairwise_by_slice: dict[str, dict[str, dict[str, float | int]]] = {}
    test_pairwise_by_slice: dict[str, dict[str, dict[str, float | int]]] = {}
    for slice_name in slice_names:
        validation_pairwise_by_slice[slice_name] = merged_slice_rows(
            pairwise_accuracy_by_slice(pairwise_rows, candidates, predictions, "validation", "baseline", slice_name),
            pairwise_accuracy_by_slice(pairwise_rows, candidates, predictions, "validation", "learned", slice_name),
        )
        test_pairwise_by_slice[slice_name] = merged_slice_rows(
            pairwise_accuracy_by_slice(pairwise_rows, candidates, predictions, "test", "baseline", slice_name),
            pairwise_accuracy_by_slice(pairwise_rows, candidates, predictions, "test", "learned", slice_name),
        )

    validation_by_source = validation_pairwise_by_slice["source_group"]
    test_by_source = test_pairwise_by_slice["source_group"]

    val_pair_b, val_pair_count = pairwise_accuracy(pairwise_rows, candidates, predictions, "validation", "baseline")
    val_pair_l, _ = pairwise_accuracy(pairwise_rows, candidates, predictions, "validation", "learned")
    test_pair_b, test_pair_count = pairwise_accuracy(pairwise_rows, candidates, predictions, "test", "baseline")
    test_pair_l, _ = pairwise_accuracy(pairwise_rows, candidates, predictions, "test", "learned")

    findings: list[str] = []
    if val_pair_l is not None and val_pair_b is not None and val_pair_l > val_pair_b:
        findings.append(
            f"Learned ranker improves validation pairwise accuracy from {val_pair_b:.3f} to {val_pair_l:.3f}, which is a cleaner signal than NDCG on the tiny validation groups."
        )
    if test_pair_l is not None and test_pair_b is not None and test_pair_l < test_pair_b:
        findings.append(
            f"Test pairwise accuracy regresses slightly from {test_pair_b:.3f} to {test_pair_l:.3f}; the model is not yet uniformly better outside validation."
        )
    if metrics["group_counts"]["validation_min3"] <= 3:
        findings.append(
            "Validation ranking groups with size >= 3 are still only 3, so NDCG is close to saturated and not strong enough as the sole promotion gate."
        )

    weak_test_slices = []
    for source, row in test_by_source.items():
        if row["count"] >= 3 and row["learned_accuracy"] < row["baseline_accuracy"]:
            weak_test_slices.append((source, row["baseline_accuracy"], row["learned_accuracy"], row["count"]))
    for source, baseline_acc, learned_acc, count in weak_test_slices[:4]:
        findings.append(
            f"On test pairwise comparisons, slice `{source}` regresses from {baseline_acc:.3f} to {learned_acc:.3f} over {count} rows and should be reviewed before changing runtime ranking."
        )

    if "translated_from_russian" in test_pairwise_by_slice["origin_language_slice"]:
        row = test_pairwise_by_slice["origin_language_slice"]["translated_from_russian"]
        findings.append(
            f"Translated-origin test pairs remain small ({row['count']}), so multilingual conclusions are still weak even after v7."
        )

    summary = {
        "row_count": len(candidates),
        "overall": {
            "validation_ndcg_at_3_baseline": metrics["baseline"]["validation_ndcg_at_3"],
            "validation_ndcg_at_3_learned": metrics["learned"]["validation_ndcg_at_3"],
            "validation_pairwise_accuracy_baseline": val_pair_b,
            "validation_pairwise_accuracy_learned": val_pair_l,
            "test_pairwise_accuracy_baseline": test_pair_b,
            "test_pairwise_accuracy_learned": test_pair_l,
            "test_ndcg_at_3_baseline": mean_ndcg(test_groups_min3, predictions, "baseline", 3),
            "test_ndcg_at_3_learned": mean_ndcg(test_groups_min3, predictions, "learned", 3),
            "validation_shortlist_recall_at_3_baseline": shortlist_recall_at_k(validation_groups, batches, candidates, predictions, "validation", "baseline", 3)[0],
            "validation_shortlist_recall_at_3_learned": shortlist_recall_at_k(validation_groups, batches, candidates, predictions, "validation", "learned", 3)[0],
            "test_shortlist_recall_at_3_baseline": shortlist_recall_at_k(test_groups_min3, batches, candidates, predictions, "test", "baseline", 3)[0],
            "test_shortlist_recall_at_3_learned": shortlist_recall_at_k(test_groups_min3, batches, candidates, predictions, "test", "learned", 3)[0],
        },
        "validation_pairwise_by_source": validation_by_source,
        "test_pairwise_by_source": test_by_source,
        "validation_pairwise_by_slice": validation_pairwise_by_slice,
        "test_pairwise_by_slice": test_pairwise_by_slice,
        "disagreements": disagreement_examples(pairwise_rows, candidates, predictions, "validation"),
        "findings": findings,
        "group_counts": metrics["group_counts"],
        "pairwise_counts": {
            "validation": val_pair_count,
            "test": test_pair_count,
        },
    }
    write_report(summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
