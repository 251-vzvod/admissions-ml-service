from __future__ import annotations

import csv
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = ROOT / "data" / "ml_workbench"
LABELS_DIR = DATA_ROOT / "labels"
EXPORTS_DIR = DATA_ROOT / "exports"

INDIVIDUAL_CSV = LABELS_DIR / "human_labels_individual_llm_v2.csv"
ADJUDICATED_CSV = LABELS_DIR / "human_labels_adjudicated.csv"
PAIRWISE_CSV = LABELS_DIR / "pairwise_labels.csv"
BATCH_JSONL = LABELS_DIR / "batch_shortlist_tasks.jsonl"

SOURCE_JSONLS = {
    "seed_pack": DATA_ROOT / "processed" / "english_candidates_api_input_v1.jsonl",
    "synthetic_batch_v1": DATA_ROOT / "raw" / "generated" / "batch_v1" / "synthetic_batch_v1_api_input.jsonl",
    "contrastive_batch_v2": DATA_ROOT / "raw" / "generated" / "contrastive_batch_v2" / "contrastive_batch_v2_api_input.jsonl",
    "translated_batch_v3": DATA_ROOT / "raw" / "generated" / "translated_batch_v3" / "translated_batch_v3_api_input.jsonl",
    "messy_batch_v4": DATA_ROOT / "raw" / "generated" / "messy_batch_v4" / "messy_batch_v4_api_input.jsonl",
    "messy_batch_v5": DATA_ROOT / "raw" / "generated" / "messy_batch_v5" / "messy_batch_v5_api_input.jsonl",
    "messy_batch_v5_extension": DATA_ROOT / "raw" / "generated" / "messy_batch_v5_extension" / "messy_batch_v5_extension_api_input.jsonl",
    "ordinary_batch_v6": DATA_ROOT / "raw" / "generated" / "ordinary_batch_v6" / "ordinary_batch_v6_api_input.jsonl",
}

TRAINING_JSONL = EXPORTS_DIR / "training_dataset_v2.jsonl"
TRAINING_CSV = EXPORTS_DIR / "training_dataset_v2.csv"
MANIFEST_JSON = EXPORTS_DIR / "training_dataset_v2_manifest.json"

WORD_RE = re.compile(r"\b\w+\b")
SPLIT_SEED = 20260402


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


def source_group_for_id(candidate_id: str) -> str:
    if candidate_id.startswith("cand_"):
        return "seed_pack"
    if candidate_id.startswith("syn_eng_v1_"):
        return "synthetic_batch_v1"
    if candidate_id.startswith("syn_contrast_v2_"):
        return "contrastive_batch_v2"
    if candidate_id.startswith("tr_ru_v3_"):
        return "translated_batch_v3"
    if candidate_id.startswith("syn_messy_v4_"):
        return "messy_batch_v4"
    if candidate_id.startswith("syn_ord_v6_"):
        return "ordinary_batch_v6"
    if candidate_id.startswith("syn_messy_v5_"):
        suffix = candidate_id.rsplit("_", 1)[-1]
        try:
            ordinal = int(suffix)
        except ValueError as exc:
            raise ValueError(f"Unrecognized messy_batch_v5 candidate id: {candidate_id}") from exc
        return "messy_batch_v5_extension" if ordinal >= 61 else "messy_batch_v5"
    raise ValueError(f"Unrecognized candidate_id prefix: {candidate_id}")


def origin_language_slice(source_group: str) -> str:
    if source_group == "translated_batch_v3":
        return "translated_from_russian"
    return "english_direct"


def total_word_count(record: dict[str, Any]) -> int:
    text_inputs = record["text_inputs"]
    parts: list[str] = [text_inputs.get("motivation_letter_text") or ""]
    for item in text_inputs.get("motivation_questions") or []:
        parts.append(item.get("question") or "")
        parts.append(item.get("answer") or "")
    parts.append(text_inputs.get("interview_text") or "")
    return len(WORD_RE.findall(" ".join(parts)))


def joined_questions_text(record: dict[str, Any]) -> str:
    items = record["text_inputs"].get("motivation_questions") or []
    return "\n\n".join(f"Q: {item.get('question', '')}\nA: {item.get('answer', '')}" for item in items)


def load_candidate_pool() -> dict[str, dict[str, Any]]:
    pool: dict[str, dict[str, Any]] = {}
    for path in SOURCE_JSONLS.values():
        for record in load_jsonl(path):
            candidate_id = record["candidate_id"]
            if candidate_id in pool:
                raise ValueError(f"Duplicate candidate_id in source pools: {candidate_id}")
            pool[candidate_id] = record
    return pool


def load_individual_rows() -> dict[str, dict[str, str]]:
    with INDIVIDUAL_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping: dict[str, dict[str, str]] = {}
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id in mapping:
            raise ValueError(f"Duplicate candidate_id in individual CSV: {candidate_id}")
        mapping[candidate_id] = row
    return mapping


def load_adjudicated_rows() -> dict[str, dict[str, str]]:
    with ADJUDICATED_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    mapping: dict[str, dict[str, str]] = {}
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id in mapping:
            raise ValueError(f"Duplicate candidate_id in adjudicated CSV: {candidate_id}")
        mapping[candidate_id] = row
    return mapping


def build_split_map(candidate_ids: list[str]) -> dict[str, str]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for candidate_id in candidate_ids:
        grouped[source_group_for_id(candidate_id)].append(candidate_id)

    rng = random.Random(SPLIT_SEED)
    split_map: dict[str, str] = {}
    for source_group, ids in grouped.items():
        shuffled = sorted(ids)
        rng.shuffle(shuffled)
        total = len(shuffled)
        train_count = round(total * 0.70)
        val_count = round(total * 0.15)
        if total >= 3:
            train_count = min(train_count, total - 2)
            val_count = min(val_count, total - train_count - 1)
        for idx, candidate_id in enumerate(shuffled):
            if idx < train_count:
                split_map[candidate_id] = "train"
            elif idx < train_count + val_count:
                split_map[candidate_id] = "validation"
            else:
                split_map[candidate_id] = "test"
    return split_map


def pairwise_count() -> int:
    with PAIRWISE_CSV.open("r", encoding="utf-8-sig", newline="") as handle:
        return sum(1 for _ in csv.DictReader(handle))


def batch_count() -> int:
    with BATCH_JSONL.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def build_rows() -> list[dict[str, Any]]:
    pool = load_candidate_pool()
    individual = load_individual_rows()
    adjudicated = load_adjudicated_rows()

    candidate_ids = sorted(individual)
    if set(candidate_ids) != set(adjudicated):
        raise ValueError("Individual and adjudicated candidate_id sets do not match.")
    if set(candidate_ids) != set(pool):
        missing_in_pool = sorted(set(candidate_ids) - set(pool))
        extra_in_pool = sorted(set(pool) - set(candidate_ids))
        raise ValueError(
            f"Candidate pool mismatch. Missing in pool: {missing_in_pool[:10]} Extra in pool: {extra_in_pool[:10]}"
        )

    split_map = build_split_map(candidate_ids)
    rows: list[dict[str, Any]] = []
    for candidate_id in candidate_ids:
        payload = pool[candidate_id]
        individual_row = individual[candidate_id]
        adjudicated_row = adjudicated[candidate_id]
        source_group = source_group_for_id(candidate_id)
        education = payload["structured_data"]["education"]
        behavioral = payload["behavioral_signals"]
        text_inputs = payload["text_inputs"]
        rows.append(
            {
                "candidate_id": candidate_id,
                "source_group": source_group,
                "origin_language_slice": origin_language_slice(source_group),
                "split": split_map[candidate_id],
                "english_type": education["english_proficiency"]["type"],
                "english_score": education["english_proficiency"]["score"],
                "school_certificate_type": education["school_certificate"]["type"],
                "school_certificate_score": education["school_certificate"]["score"],
                "completion_rate": behavioral.get("completion_rate"),
                "returned_to_edit": behavioral.get("returned_to_edit"),
                "skipped_optional_questions": behavioral.get("skipped_optional_questions"),
                "text_length_bucket": individual_row["text_length_bucket"],
                "has_interview_text": parse_bool(individual_row["has_interview_text"]),
                "has_transcript": parse_bool(individual_row["has_transcript"]),
                "motivation_question_count": len(text_inputs.get("motivation_questions") or []),
                "total_text_word_count": total_word_count(payload),
                "motivation_letter_text": text_inputs.get("motivation_letter_text") or "",
                "motivation_questions_text": joined_questions_text(payload),
                "interview_text": text_inputs.get("interview_text") or "",
                "final_recommendation": adjudicated_row["final_recommendation"],
                "final_committee_priority": int(adjudicated_row["final_committee_priority"]),
                "final_shortlist_band": parse_bool(adjudicated_row["final_shortlist_band"]),
                "final_hidden_potential_band": parse_bool(adjudicated_row["final_hidden_potential_band"]),
                "final_support_needed_band": parse_bool(adjudicated_row["final_support_needed_band"]),
                "final_authenticity_review_band": parse_bool(adjudicated_row["final_authenticity_review_band"]),
                "final_notes": adjudicated_row["final_notes"],
                "reviewer_confidence": int(individual_row["reviewer_confidence"]),
                "review_round": individual_row["review_round"],
            }
        )
    return rows


def write_outputs(rows: list[dict[str, Any]]) -> None:
    EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
    with TRAINING_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    fieldnames = list(rows[0].keys())
    with TRAINING_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_manifest(rows: list[dict[str, Any]]) -> None:
    split_counts = Counter(row["split"] for row in rows)
    source_counts = Counter(row["source_group"] for row in rows)
    split_source_counts: dict[str, dict[str, int]] = {}
    for split in sorted(split_counts):
        split_source_counts[split] = dict(Counter(row["source_group"] for row in rows if row["split"] == split))
    recommendation_counts = dict(Counter(row["final_recommendation"] for row in rows))
    shortlist_counts = dict(Counter(str(row["final_shortlist_band"]).lower() for row in rows))

    manifest = {
        "dataset_name": "training_dataset_v2",
        "generated_from": {
            "individual_labels": str(INDIVIDUAL_CSV),
            "adjudicated_labels": str(ADJUDICATED_CSV),
            "pairwise_labels": str(PAIRWISE_CSV),
            "batch_shortlist_tasks": str(BATCH_JSONL),
            **{f"{name}_payloads": str(path) for name, path in SOURCE_JSONLS.items()},
        },
        "row_count": len(rows),
        "split_seed": SPLIT_SEED,
        "split_counts": dict(split_counts),
        "source_counts": dict(source_counts),
        "split_source_counts": split_source_counts,
        "recommendation_counts": recommendation_counts,
        "shortlist_band_counts": shortlist_counts,
        "pairwise_row_count": pairwise_count(),
        "batch_shortlist_task_count": batch_count(),
        "notes": [
            "Candidate-level canonical training table after adding messy realism batches and ordinary disagreement batch.",
            "This export joins public input payloads with current bootstrap adjudicated labels.",
            "Pairwise and batch artifacts are sibling supervision sources and are not flattened into each candidate row.",
            "Translated batch v3 is marked as origin_language_slice=translated_from_russian.",
        ],
    }
    with MANIFEST_JSON.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def main() -> None:
    rows = build_rows()
    write_outputs(rows)
    write_manifest(rows)
    print(
        json.dumps(
            {
                "row_count": len(rows),
                "jsonl": str(TRAINING_JSONL),
                "csv": str(TRAINING_CSV),
                "manifest": str(MANIFEST_JSON),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
