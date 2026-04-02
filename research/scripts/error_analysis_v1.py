from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from app.services.offline_ranker import build_offline_ranker_feature_map
from app.services.pipeline import ScoringPipeline
from research.scripts.train_shortlist_ranker_v1 import (
    FEATURE_VARIANT_NAME,
    LEARNED_BLEND_ALPHA,
    MODEL_DIR,
    PAIRWISE_CSV,
    TRAINING_CSV,
    load_payloads,
)


REPORT_MD = MODEL_DIR / "error_analysis_report.md"
REPORT_JSON = MODEL_DIR / "error_analysis_summary.json"
CURRENT_PREDICTIONS_CSV = MODEL_DIR / "candidate_predictions.csv"
MONOTONE_PROBE_DIR = ROOT / "data" / "ml_workbench" / "exports" / "models" / "probe_drop_support_monotone"
MONOTONE_PREDICTIONS_CSV = MONOTONE_PROBE_DIR / "candidate_predictions.csv"


@dataclass(frozen=True)
class CandidateMeta:
    candidate_id: str
    split: str
    source_group: str
    recommendation: str
    committee_priority: int
    shortlist_band: bool
    hidden_potential_band: bool
    support_needed_band: bool
    authenticity_review_band: bool
    has_interview_text: bool
    text_length_bucket: str
    reviewer_confidence: int


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() == "true"


def load_training_meta() -> dict[str, CandidateMeta]:
    rows: dict[str, CandidateMeta] = {}
    with TRAINING_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            candidate_id = row["candidate_id"]
            rows[candidate_id] = CandidateMeta(
                candidate_id=candidate_id,
                split=row["split"],
                source_group=row["source_group"],
                recommendation=row["final_recommendation"],
                committee_priority=int(row["final_committee_priority"]),
                shortlist_band=parse_bool(row["final_shortlist_band"]),
                hidden_potential_band=parse_bool(row["final_hidden_potential_band"]),
                support_needed_band=parse_bool(row["final_support_needed_band"]),
                authenticity_review_band=parse_bool(row["final_authenticity_review_band"]),
                has_interview_text=parse_bool(row["has_interview_text"]),
                text_length_bucket=row["text_length_bucket"],
                reviewer_confidence=int(row["reviewer_confidence"]),
            )
    return rows


def load_predictions(path: Path) -> dict[str, dict[str, float]]:
    predictions: dict[str, dict[str, float]] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            predictions[row["candidate_id"]] = {
                "baseline": float(row["baseline_score"]),
                "learned": float(row["learned_score"]),
            }
    return predictions


def load_pairwise_rows() -> list[dict[str, str]]:
    with PAIRWISE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def correct_prediction(predictions: dict[str, dict[str, float]], left: str, right: str, preferred: str, model_key: str) -> bool:
    predicted = left if predictions[left][model_key] >= predictions[right][model_key] else right
    return predicted == preferred


def score_gap(predictions: dict[str, dict[str, float]], preferred: str, other: str, model_key: str) -> float:
    return predictions[preferred][model_key] - predictions[other][model_key]


def build_pair_record(
    row: dict[str, str],
    candidate_meta: dict[str, CandidateMeta],
    predictions: dict[str, dict[str, float]],
) -> dict[str, Any]:
    left = row["candidate_id_left"]
    right = row["candidate_id_right"]
    preferred = row["preferred_candidate_id"]
    other = right if preferred == left else left
    preferred_meta = candidate_meta[preferred]
    other_meta = candidate_meta[other]
    return {
        "pair_id": row["pair_id"],
        "split": preferred_meta.split,
        "reason_primary": row["reason_primary"],
        "preference_strength": int(row["preference_strength"]),
        "preferred_candidate_id": preferred,
        "other_candidate_id": other,
        "preferred_source_group": preferred_meta.source_group,
        "other_source_group": other_meta.source_group,
        "preferred_recommendation": preferred_meta.recommendation,
        "other_recommendation": other_meta.recommendation,
        "preferred_support_needed": preferred_meta.support_needed_band,
        "preferred_authenticity_review": preferred_meta.authenticity_review_band,
        "preferred_hidden_potential": preferred_meta.hidden_potential_band,
        "preferred_has_interview_text": preferred_meta.has_interview_text,
        "preferred_text_length_bucket": preferred_meta.text_length_bucket,
        "baseline_correct": correct_prediction(predictions, left, right, preferred, "baseline"),
        "learned_correct": correct_prediction(predictions, left, right, preferred, "learned"),
        "baseline_gap_preferred_minus_other": score_gap(predictions, preferred, other, "baseline"),
        "learned_gap_preferred_minus_other": score_gap(predictions, preferred, other, "learned"),
    }


def feature_snapshot(candidate_ids: list[str]) -> dict[str, dict[str, Any]]:
    payloads = load_payloads()
    pipeline = ScoringPipeline()
    snapshots: dict[str, dict[str, Any]] = {}
    for candidate_id in sorted(set(candidate_ids)):
        result = pipeline.score_candidate(payloads[candidate_id], enable_llm_explainability=False)
        fmap = build_offline_ranker_feature_map(result)
        snapshots[candidate_id] = {
            "recommendation": result.recommendation,
            "merit_score": getattr(result, "merit_score", 0),
            "confidence_score": getattr(result, "confidence_score", 0),
            "authenticity_risk": getattr(result, "authenticity_risk", 0),
            "hidden_potential_score": getattr(result, "hidden_potential_score", 0),
            "support_needed_score": getattr(result, "support_needed_score", 0),
            "shortlist_priority_score": getattr(result, "shortlist_priority_score", 0),
            "evidence_coverage_score": getattr(result, "evidence_coverage_score", 0),
            "trajectory_score": getattr(result, "trajectory_score", 0),
            "feature_map": {key: round(float(value), 4) for key, value in fmap.items()},
        }
    return snapshots


def summarize_case_group(pair_rows: list[dict[str, Any]]) -> dict[str, Any]:
    reason_counts: dict[str, int] = {}
    learned_wrong = [row for row in pair_rows if not row["learned_correct"]]
    baseline_wrong = [row for row in pair_rows if not row["baseline_correct"]]
    for row in pair_rows:
        reason_counts[row["reason_primary"]] = reason_counts.get(row["reason_primary"], 0) + 1
    involved_wrong: dict[str, int] = {}
    for row in learned_wrong:
        involved_wrong[row["preferred_candidate_id"]] = involved_wrong.get(row["preferred_candidate_id"], 0) + 1
        involved_wrong[row["other_candidate_id"]] = involved_wrong.get(row["other_candidate_id"], 0) + 1
    return {
        "count": len(pair_rows),
        "baseline_accuracy": mean(row["baseline_correct"] for row in pair_rows) if pair_rows else None,
        "learned_accuracy": mean(row["learned_correct"] for row in pair_rows) if pair_rows else None,
        "reason_counts": reason_counts,
        "learned_wrong_count": len(learned_wrong),
        "baseline_wrong_count": len(baseline_wrong),
        "learned_wrong_pairs": learned_wrong,
        "baseline_wrong_pairs": baseline_wrong,
        "most_involved_candidates_in_learned_wrong": sorted(
            [{"candidate_id": cid, "count": count} for cid, count in involved_wrong.items()],
            key=lambda item: (-item["count"], item["candidate_id"]),
        )[:8],
    }


def compare_probe(
    pair_rows: list[dict[str, Any]],
    candidate_meta: dict[str, CandidateMeta],
) -> dict[str, Any] | None:
    if not MONOTONE_PREDICTIONS_CSV.exists():
        return None
    monotone_predictions = load_predictions(MONOTONE_PREDICTIONS_CSV)
    computed: list[dict[str, Any]] = []
    for row in pair_rows:
        pair_id = row["pair_id"]
        preferred = row["preferred_candidate_id"]
        other = row["other_candidate_id"]
        preferred_meta = candidate_meta[preferred]
        left = preferred if preferred in {preferred, other} else preferred
        # Reconstruct original ordering is unnecessary for pairwise correctness if we compare preferred and other directly.
        predicted = preferred if monotone_predictions[preferred]["learned"] >= monotone_predictions[other]["learned"] else other
        computed.append(
            {
                "pair_id": pair_id,
                "preferred_candidate_id": preferred,
                "other_candidate_id": other,
                "monotone_correct": predicted == preferred,
                "monotone_gap_preferred_minus_other": monotone_predictions[preferred]["learned"] - monotone_predictions[other]["learned"],
                "split": preferred_meta.split,
            }
        )
    accuracy = mean(item["monotone_correct"] for item in computed) if computed else None
    return {
        "accuracy": accuracy,
        "pair_count": len(computed),
        "pair_rows": computed,
    }


def markdown_for_pairs(title: str, summary: dict[str, Any]) -> list[str]:
    lines = [
        f"## {title}",
        "",
        f"- count: `{summary['count']}`",
        f"- baseline accuracy: `{summary['baseline_accuracy']:.4f}`" if summary["baseline_accuracy"] is not None else "- baseline accuracy: `n/a`",
        f"- learned accuracy: `{summary['learned_accuracy']:.4f}`" if summary["learned_accuracy"] is not None else "- learned accuracy: `n/a`",
        f"- reason_counts: `{summary['reason_counts']}`",
        "",
        "### Learned Wrong Pairs",
    ]
    if not summary["learned_wrong_pairs"]:
        lines.append("- none")
    else:
        for row in summary["learned_wrong_pairs"]:
            lines.append(
                "- "
                f"`{row['pair_id']}` preferred `{row['preferred_candidate_id']}` over `{row['other_candidate_id']}`; "
                f"reason `{row['reason_primary']}`, strength `{row['preference_strength']}`, "
                f"baseline_gap `{row['baseline_gap_preferred_minus_other']:.4f}`, learned_gap `{row['learned_gap_preferred_minus_other']:.4f}`"
            )
    lines.extend(["", "### Baseline Wrong Pairs"])
    if not summary["baseline_wrong_pairs"]:
        lines.append("- none")
    else:
        for row in summary["baseline_wrong_pairs"]:
            lines.append(
                "- "
                f"`{row['pair_id']}` preferred `{row['preferred_candidate_id']}` over `{row['other_candidate_id']}`; "
                f"reason `{row['reason_primary']}`, strength `{row['preference_strength']}`, "
                f"baseline_gap `{row['baseline_gap_preferred_minus_other']:.4f}`, learned_gap `{row['learned_gap_preferred_minus_other']:.4f}`"
            )
    return lines


def main() -> None:
    candidate_meta = load_training_meta()
    predictions = load_predictions(CURRENT_PREDICTIONS_CSV)
    pairwise_rows = load_pairwise_rows()

    messy_extension_pairs = [
        build_pair_record(row, candidate_meta, predictions)
        for row in pairwise_rows
        if candidate_meta[row["candidate_id_left"]].split == "test"
        and candidate_meta[row["candidate_id_right"]].split == "test"
        if candidate_meta[row["preferred_candidate_id"]].split == "test"
        and candidate_meta[row["preferred_candidate_id"]].source_group == "messy_batch_v5_extension"
    ]
    manual_review_pairs = [
        build_pair_record(row, candidate_meta, predictions)
        for row in pairwise_rows
        if candidate_meta[row["candidate_id_left"]].split in {"validation", "test"}
        and candidate_meta[row["candidate_id_right"]].split in {"validation", "test"}
        and candidate_meta[row["candidate_id_left"]].split == candidate_meta[row["candidate_id_right"]].split
        if candidate_meta[row["preferred_candidate_id"]].split in {"validation", "test"}
        and candidate_meta[row["preferred_candidate_id"]].recommendation == "manual_review_required"
    ]

    messy_summary = summarize_case_group(messy_extension_pairs)
    manual_summary = summarize_case_group(manual_review_pairs)

    involved_candidate_ids = {
        row["preferred_candidate_id"]
        for row in (messy_summary["learned_wrong_pairs"] + manual_summary["learned_wrong_pairs"] + messy_summary["baseline_wrong_pairs"])
    }
    involved_candidate_ids.update(
        row["other_candidate_id"]
        for row in (messy_summary["learned_wrong_pairs"] + manual_summary["learned_wrong_pairs"] + manual_summary["baseline_wrong_pairs"])
    )
    candidate_snapshots = feature_snapshot(sorted(involved_candidate_ids))

    monotone_messy = compare_probe(messy_extension_pairs, candidate_meta)
    monotone_manual = compare_probe(manual_review_pairs, candidate_meta)

    findings = [
        (
            "Current shortlist model "
            f"`{FEATURE_VARIANT_NAME}` with blend alpha `{LEARNED_BLEND_ALPHA}` "
            f"gets `{messy_summary['learned_wrong_count']}` learned-wrong pairs out of `{messy_summary['count']}` on "
            "`messy_batch_v5_extension` test comparisons."
        ),
        (
            "On manual-review positives across validation/test, learned accuracy is "
            f"`{manual_summary['learned_accuracy']:.4f}` vs baseline `{manual_summary['baseline_accuracy']:.4f}` "
            f"over `{manual_summary['count']}` pairs."
        ),
        (
            "The remaining shortlist weakness is narrow and local: most unresolved misses still cluster around "
            "manual-review routing or trajectory-heavy messy-v5-extension comparisons."
        ),
    ]
    if monotone_messy is not None:
        findings.append(
            "A monotone LightGBM probe remains available for comparison, but it is still a side experiment and not the promoted shortlist model."
        )

    recommendations = [
        "Do not push the shortlist ranker to absorb manual-review routing logic. Train a separate Phase 4 manual-review / confidence model.",
        "Add a tiny targeted pairwise pack around `messy_batch_v5_extension` and manual-review positives instead of generating another broad batch.",
        "Keep the current `drop_support` ranker as the best overall shortlist model, but treat `manual_review_required` as a sidecar decision problem.",
        "If the next aggregate-feature iteration is attempted, test monotonic constraints only after adding more no-interview and manual-review supervision.",
    ]

    summary = {
        "model_dir": str(MODEL_DIR),
        "current_model_variant": FEATURE_VARIANT_NAME,
        "learned_blend_alpha": LEARNED_BLEND_ALPHA,
        "messy_batch_v5_extension_test": messy_summary,
        "manual_review_required_validation_test": manual_summary,
        "candidate_snapshots": candidate_snapshots,
        "monotone_probe": {
            "available": monotone_messy is not None,
            "messy_batch_v5_extension_test": monotone_messy,
            "manual_review_required_validation_test": monotone_manual,
        },
        "findings": findings,
        "recommendations": recommendations,
    }

    with REPORT_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    lines = [
        "# Error Analysis Report",
        "",
        f"- model_dir: `{MODEL_DIR}`",
        f"- current_model_variant: `{FEATURE_VARIANT_NAME}`",
        f"- learned_blend_alpha: `{LEARNED_BLEND_ALPHA}`",
        "",
    ]
    lines.extend(markdown_for_pairs("Messy Batch V5 Extension Test", messy_summary))
    lines.extend([""])
    lines.extend(markdown_for_pairs("Manual Review Required Validation/Test", manual_summary))
    lines.extend(["", "## Candidate Snapshots", ""])
    for candidate_id in sorted(candidate_snapshots):
        snapshot = candidate_snapshots[candidate_id]
        lines.append(
            f"- `{candidate_id}`: recommendation `{snapshot['recommendation']}`, "
            f"merit `{snapshot['merit_score']}`, confidence `{snapshot['confidence_score']}`, "
            f"auth_risk `{snapshot['authenticity_risk']}`, hidden `{snapshot['hidden_potential_score']}`, "
            f"support `{snapshot['support_needed_score']}`, trajectory `{snapshot['trajectory_score']}`"
        )
    if monotone_messy is not None and monotone_manual is not None:
        lines.extend(
            [
                "",
                "## Monotone Probe",
                "",
                f"- messy_batch_v5_extension test accuracy: `{monotone_messy['accuracy']:.4f}` over `{monotone_messy['pair_count']}` pairs",
                f"- manual_review validation/test accuracy: `{monotone_manual['accuracy']:.4f}` over `{monotone_manual['pair_count']}` pairs",
            ]
        )
    lines.extend(["", "## Findings", ""])
    for finding in findings:
        lines.append(f"- {finding}")
    lines.extend(["", "## Recommendations", ""])
    for recommendation in recommendations:
        lines.append(f"- {recommendation}")

    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
