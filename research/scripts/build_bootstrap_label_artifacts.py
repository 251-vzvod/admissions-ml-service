from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
LABELS_DIR = ROOT / "data" / "ml_workbench" / "labels"
INDIVIDUAL_CSV = LABELS_DIR / "human_labels_individual_llm_v2.csv"
ADJUDICATED_CSV = LABELS_DIR / "human_labels_adjudicated.csv"
PAIRWISE_CSV = LABELS_DIR / "pairwise_labels.csv"
BATCH_JSONL = LABELS_DIR / "batch_shortlist_tasks.jsonl"
SUMMARY_MD = LABELS_DIR / "bootstrap_label_artifacts_summary.md"


@dataclass(frozen=True)
class CandidateLabel:
    candidate_id: str
    source_group: str
    recommendation: str
    committee_priority: int
    shortlist_band: bool
    hidden_potential_band: bool
    support_needed_band: bool
    authenticity_review_band: bool
    reviewer_confidence: int
    notes: str
    text_length_bucket: str
    has_interview_text: bool


RECOMMENDATION_WEIGHT = {
    "review_priority": 5,
    "standard_review": 4,
    "manual_review_required": 3,
    "insufficient_evidence": 2,
    "incomplete_application": 1,
    "invalid": 0,
}

TARGETED_BATCH_SPECS = [
    {
        "batch_id": "targeted_batch_001",
        "candidate_ids": [
            "syn_gap_v7_013",
            "syn_gap_v7_036",
            "syn_gap_v7_017",
            "syn_gap_v7_035",
            "syn_gap_v7_003",
            "syn_gap_v7_041",
            "syn_gap_v7_050",
            "syn_messy_v5_006",
        ],
        "notes": "Targeted shortlist calibration batch mixing train and validation no-interview/support-needed cases around subtle manual-review routing.",
    },
    {
        "batch_id": "targeted_batch_002",
        "candidate_ids": [
            "syn_gap_v7_002",
            "syn_gap_v7_006",
            "syn_messy_v5_050",
            "syn_messy_v5_059",
            "syn_gap_v7_001",
            "syn_gap_v7_015",
            "syn_messy_v5_010",
            "syn_messy_v5_044",
        ],
        "notes": "Targeted shortlist calibration batch mixing train and validation subtle review-risk positives against close support-needed standards.",
    },
    {
        "batch_id": "targeted_batch_003",
        "candidate_ids": [
            "syn_messy_v5_033",
            "syn_messy_v5_040",
            "syn_messy_v5_064",
            "syn_messy_v5_069",
            "syn_gap_v7_071",
            "syn_messy_v5_030",
            "syn_messy_v5_074",
            "syn_messy_v5_076",
        ],
        "notes": "Targeted shortlist calibration batch mixing train and test shortlist / hidden-potential disagreements around messy_batch_v5_extension.",
    },
    {
        "batch_id": "targeted_batch_004",
        "candidate_ids": [
            "syn_gap_v7_004",
            "syn_gap_v7_012",
            "syn_gap_v7_014",
            "syn_messy_v5_060",
            "syn_gap_v7_037",
            "syn_gap_v7_071",
            "syn_messy_v5_017",
            "syn_messy_v5_031",
        ],
        "notes": "Targeted shortlist calibration batch mixing train and test borderline cases where ordinary standards compete with stronger but messier profiles.",
    },
    {
        "batch_id": "targeted_batch_005",
        "candidate_ids": [
            "syn_eng_v1_045",
            "syn_eng_v1_047",
            "syn_gap_v7_005",
            "syn_gap_v7_010",
            "syn_eng_v1_048",
            "syn_gap_v7_018",
            "syn_gap_v7_068",
            "syn_gap_v7_070",
        ],
        "notes": "Targeted shortlist calibration batch mixing train and validation low-priority authenticity-review and no-interview cases.",
    },
    {
        "batch_id": "targeted_batch_006",
        "candidate_ids": [
            "syn_contrast_v2_024",
            "syn_gap_v7_009",
            "syn_gap_v7_042",
            "syn_gap_v7_006",
            "syn_contrast_v2_013",
            "syn_eng_v1_023",
            "syn_ord_v6_020",
            "syn_messy_v5_058",
        ],
        "notes": "Targeted shortlist calibration batch mixing train and validation cases where authenticity-review positives compete against shortlist and support-needed standards.",
    },
]


def now_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_bool(value: str) -> bool:
    return value.strip().lower() == "true"


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
    if candidate_id.startswith("syn_gap_v7_"):
        return "gap_fill_batch_v7"
    if candidate_id.startswith("syn_messy_v5_"):
        suffix = candidate_id.rsplit("_", 1)[-1]
        try:
            ordinal = int(suffix)
        except ValueError:
            return "messy_batch_v5"
        return "messy_batch_v5_extension" if ordinal >= 61 else "messy_batch_v5"
    return "unknown"


def load_candidates() -> list[CandidateLabel]:
    with INDIVIDUAL_CSV.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    seen: set[str] = set()
    candidates: list[CandidateLabel] = []
    for row in rows:
        candidate_id = row["candidate_id"]
        if candidate_id in seen:
            raise ValueError(f"Duplicate candidate_id in individual labels: {candidate_id}")
        seen.add(candidate_id)
        candidates.append(
            CandidateLabel(
                candidate_id=candidate_id,
                source_group=source_group_for_id(candidate_id),
                recommendation=row["recommendation"],
                committee_priority=int(row["committee_priority"]),
                shortlist_band=parse_bool(row["shortlist_band"]),
                hidden_potential_band=parse_bool(row["hidden_potential_band"]),
                support_needed_band=parse_bool(row["support_needed_band"]),
                authenticity_review_band=parse_bool(row["authenticity_review_band"]),
                reviewer_confidence=int(row["reviewer_confidence"]),
                notes=row["notes"].strip(),
                text_length_bucket=row["text_length_bucket"].strip(),
                has_interview_text=parse_bool(row["has_interview_text"]),
            )
        )
    return candidates


def candidate_rank_key(candidate: CandidateLabel) -> tuple[int, int, int, int, int, int, int, str]:
    return (
        1 if candidate.shortlist_band else 0,
        candidate.committee_priority,
        RECOMMENDATION_WEIGHT.get(candidate.recommendation, 0),
        1 if candidate.hidden_potential_band else 0,
        0 if candidate.authenticity_review_band else 1,
        1 if candidate.has_interview_text else 0,
        candidate.reviewer_confidence,
        candidate.candidate_id,
    )


def note_theme(candidate: CandidateLabel) -> str:
    text = candidate.notes.lower()
    if candidate.authenticity_review_band:
        return "authenticity_review"
    if candidate.recommendation == "insufficient_evidence" or "evidence" in text or "thin" in text:
        return "evidence"
    if any(token in text for token in ["community", "neighborhood", "classmate", "peer", "younger students", "neighbors"]):
        return "community"
    if any(token in text for token in ["built", "organized", "created", "prototype", "launched", "mentorship"]):
        return "leadership"
    if candidate.hidden_potential_band:
        return "trajectory"
    if "motivation" in text or "fit" in text:
        return "motivation"
    return "trajectory"


def write_adjudicated(candidates: list[CandidateLabel], reviewed_at: str) -> None:
    fieldnames = [
        "candidate_id",
        "adjudication_status",
        "adjudicator_id",
        "adjudicated_at_utc",
        "final_recommendation",
        "final_committee_priority",
        "final_shortlist_band",
        "final_hidden_potential_band",
        "final_support_needed_band",
        "final_authenticity_review_band",
        "final_notes",
        "reviewer_count",
        "disagreement_flag",
        "disagreement_summary",
        "evidence_span_count",
    ]
    with ADJUDICATED_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(
                {
                    "candidate_id": candidate.candidate_id,
                    "adjudication_status": "single_reviewer_only",
                    "adjudicator_id": "bootstrap_from_llm_v2",
                    "adjudicated_at_utc": reviewed_at,
                    "final_recommendation": candidate.recommendation,
                    "final_committee_priority": candidate.committee_priority,
                    "final_shortlist_band": str(candidate.shortlist_band).lower(),
                    "final_hidden_potential_band": str(candidate.hidden_potential_band).lower(),
                    "final_support_needed_band": str(candidate.support_needed_band).lower(),
                    "final_authenticity_review_band": str(candidate.authenticity_review_band).lower(),
                    "final_notes": candidate.notes,
                    "reviewer_count": 1,
                    "disagreement_flag": "false",
                    "disagreement_summary": "Bootstrapped from one llm_reviewer row; no multi-review adjudication yet.",
                    "evidence_span_count": 0,
                }
            )


def round_robin_picker(pool: list[CandidateLabel], state: dict[str, int], key: str, excluded: set[str]) -> CandidateLabel | None:
    if not pool:
        return None
    start = state.get(key, 0)
    for offset in range(len(pool)):
        idx = (start + offset) % len(pool)
        candidate = pool[idx]
        if candidate.candidate_id not in excluded:
            state[key] = (idx + 1) % len(pool)
            return candidate
    return None


def build_batches(candidates: list[CandidateLabel], reviewed_at: str) -> list[dict[str, object]]:
    rng = random.Random(20260402)
    all_candidates = list(candidates)
    rng.shuffle(all_candidates)

    high = [c for c in all_candidates if c.committee_priority >= 4 or c.recommendation == "review_priority"]
    mid = [c for c in all_candidates if c.committee_priority == 3 or c.recommendation == "standard_review"]
    low = [c for c in all_candidates if c.committee_priority <= 2 or c.recommendation == "insufficient_evidence"]
    hidden = [c for c in all_candidates if c.hidden_potential_band]
    support = [c for c in all_candidates if c.support_needed_band]
    authenticity = [c for c in all_candidates if c.authenticity_review_band]
    by_source = {
        key: [c for c in all_candidates if c.source_group == key]
        for key in sorted({candidate.source_group for candidate in all_candidates})
    }
    state: dict[str, int] = {}
    preferred_order = [
        "seed_pack",
        "synthetic_batch_v1",
        "contrastive_batch_v2",
        "translated_batch_v3",
        "gap_fill_batch_v7",
        "messy_batch_v4",
        "messy_batch_v5",
        "messy_batch_v5_extension",
        "ordinary_batch_v6",
    ]
    source_rotation = [key for key in preferred_order if key in by_source]
    batches: list[dict[str, object]] = []
    batch_count = max(18, min(48, math.ceil(len(candidates) / 6)))

    for batch_num in range(1, batch_count + 1):
        excluded: set[str] = set()
        picked: list[CandidateLabel] = []

        def add_from(pool: list[CandidateLabel], key: str, preferred_source: str | None = None) -> None:
            candidate = None
            if preferred_source is not None:
                source_pool = [c for c in pool if c.source_group == preferred_source]
                candidate = round_robin_picker(source_pool, state, f"{key}:{preferred_source}", excluded)
            if candidate is None:
                candidate = round_robin_picker(pool, state, key, excluded)
            if candidate is not None:
                picked.append(candidate)
                excluded.add(candidate.candidate_id)

        rotation_len = len(source_rotation)
        add_from(high, "high_a", source_rotation[(batch_num - 1) % rotation_len])
        add_from(high, "high_b", source_rotation[batch_num % rotation_len])
        add_from(mid, "mid_a", source_rotation[(batch_num + 1) % rotation_len])
        add_from(mid, "mid_b")
        add_from(low, "low_a")
        add_from(hidden, "hidden_a")
        add_from(support, "support_a")
        if batch_num <= len(authenticity):
            add_from(authenticity, "auth_a")
        else:
            add_from(all_candidates, "wildcard")

        while len(picked) < 8:
            add_from(all_candidates, f"fallback_{len(picked)}")

        presented = list(picked)
        rng.shuffle(presented)
        ranked = sorted(presented, key=candidate_rank_key, reverse=True)
        selected_shortlist = [c.candidate_id for c in ranked if c.shortlist_band][:3]
        if not selected_shortlist:
            selected_shortlist = [ranked[0].candidate_id]

        batches.append(
            {
                "batch_id": f"bootstrap_batch_{batch_num:03d}",
                "task_created_at_utc": reviewed_at,
                "reviewer_ids": ["bootstrap_derived"],
                "candidate_ids": [c.candidate_id for c in presented],
                "selected_shortlist_candidate_ids": selected_shortlist,
                "ranked_candidate_ids": [c.candidate_id for c in ranked],
                "hidden_potential_candidate_ids": [c.candidate_id for c in ranked if c.hidden_potential_band][:3],
                "support_needed_candidate_ids": [c.candidate_id for c in ranked if c.support_needed_band][:3],
                "authenticity_review_candidate_ids": [c.candidate_id for c in ranked if c.authenticity_review_band][:3],
                "notes": "Bootstrap batch derived from single-review adjudicated labels. Use for ranking experiments only, not as final committee ground truth.",
                "batch_size": len(presented),
                "language_mix": "english_only",
                "task_version": "bootstrap_v2",
                "adjudication_status": "derived_from_single_reviewer_bootstrap",
            }
        )
    return batches


def build_targeted_batches(candidates: list[CandidateLabel], reviewed_at: str) -> list[dict[str, object]]:
    cmap = candidate_map(candidates)
    batches: list[dict[str, object]] = []

    for spec in TARGETED_BATCH_SPECS:
        missing = [candidate_id for candidate_id in spec["candidate_ids"] if candidate_id not in cmap]
        if missing:
            raise ValueError(f"Missing candidate ids for {spec['batch_id']}: {missing}")

        presented = [cmap[candidate_id] for candidate_id in spec["candidate_ids"]]
        if len(presented) != 8:
            raise ValueError(f"{spec['batch_id']} must contain exactly 8 candidate ids.")

        ranked = sorted(presented, key=candidate_rank_key, reverse=True)
        selected_shortlist = [c.candidate_id for c in ranked if c.shortlist_band][:3]
        if not selected_shortlist:
            selected_shortlist = [ranked[0].candidate_id]

        batches.append(
            {
                "batch_id": spec["batch_id"],
                "task_created_at_utc": reviewed_at,
                "reviewer_ids": ["bootstrap_targeted"],
                "candidate_ids": [c.candidate_id for c in presented],
                "selected_shortlist_candidate_ids": selected_shortlist,
                "ranked_candidate_ids": [c.candidate_id for c in ranked],
                "hidden_potential_candidate_ids": [c.candidate_id for c in ranked if c.hidden_potential_band][:3],
                "support_needed_candidate_ids": [c.candidate_id for c in ranked if c.support_needed_band][:3],
                "authenticity_review_candidate_ids": [c.candidate_id for c in ranked if c.authenticity_review_band][:3],
                "notes": spec["notes"],
                "batch_size": len(presented),
                "language_mix": "english_only",
                "task_version": "bootstrap_targeted_v1",
                "adjudication_status": "derived_from_single_reviewer_bootstrap_targeted",
            }
        )

    return batches


def preference_strength(left: CandidateLabel, right: CandidateLabel, rank_pos: dict[str, int]) -> int:
    gap = abs(rank_pos[left.candidate_id] - rank_pos[right.candidate_id])
    priority_gap = abs(left.committee_priority - right.committee_priority)
    rec_gap = abs(RECOMMENDATION_WEIGHT[left.recommendation] - RECOMMENDATION_WEIGHT[right.recommendation])
    if gap >= 4 or priority_gap >= 2 or rec_gap >= 2:
        return 3
    if gap >= 2 or left.shortlist_band != right.shortlist_band:
        return 2
    return 1


def reason_primary(preferred: CandidateLabel, other: CandidateLabel) -> str:
    if preferred.authenticity_review_band != other.authenticity_review_band and other.authenticity_review_band:
        return "authenticity_review"
    if other.recommendation == "insufficient_evidence" or preferred.recommendation == "review_priority" and other.shortlist_band is False:
        return "evidence"
    if preferred.hidden_potential_band and not other.hidden_potential_band:
        return "trajectory"
    theme = note_theme(preferred)
    if theme in {"community", "leadership", "trajectory", "motivation", "evidence", "authenticity_review"}:
        return theme
    return "trajectory"


def candidate_map(candidates: list[CandidateLabel]) -> dict[str, CandidateLabel]:
    return {candidate.candidate_id: candidate for candidate in candidates}


def write_batches(batches: list[dict[str, object]]) -> None:
    with BATCH_JSONL.open("w", encoding="utf-8", newline="\n") as handle:
        for batch in batches:
            handle.write(json.dumps(batch, ensure_ascii=False) + "\n")


def write_pairwise(candidates: list[CandidateLabel], batches: list[dict[str, object]], reviewed_at: str) -> int:
    cmap = candidate_map(candidates)
    fieldnames = [
        "pair_id",
        "batch_id",
        "reviewer_id",
        "reviewed_at_utc",
        "candidate_id_left",
        "candidate_id_right",
        "preferred_candidate_id",
        "preference_strength",
        "reason_primary",
        "reason_notes",
        "task_type",
        "hidden_potential_preference",
        "authenticity_review_preference",
        "support_needed_preference",
    ]
    pair_count = 0
    with PAIRWISE_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()

        for batch in batches:
            ranked_ids = batch["ranked_candidate_ids"]
            rank_pos = {cid: idx for idx, cid in enumerate(ranked_ids)}
            presented_ids = batch["candidate_ids"]
            for left_id, right_id in combinations(presented_ids, 2):
                left = cmap[left_id]
                right = cmap[right_id]
                preferred = left if rank_pos[left_id] < rank_pos[right_id] else right
                other = right if preferred is left else left
                pair_count += 1
                writer.writerow(
                    {
                        "pair_id": f"bootstrap_pair_{pair_count:04d}",
                        "batch_id": batch["batch_id"],
                        "reviewer_id": "bootstrap_derived",
                        "reviewed_at_utc": reviewed_at,
                        "candidate_id_left": left_id,
                        "candidate_id_right": right_id,
                        "preferred_candidate_id": preferred.candidate_id,
                        "preference_strength": preference_strength(left, right, rank_pos),
                        "reason_primary": reason_primary(preferred, other),
                        "reason_notes": f"Derived from bootstrap batch ranking: {preferred.candidate_id} is ordered ahead of {other.candidate_id} in {batch['batch_id']}.",
                        "task_type": "shortlist_first",
                        "hidden_potential_preference": (
                            preferred.candidate_id
                            if preferred.hidden_potential_band != other.hidden_potential_band and preferred.hidden_potential_band
                            else other.candidate_id
                            if preferred.hidden_potential_band != other.hidden_potential_band and other.hidden_potential_band
                            else ""
                        ),
                        "authenticity_review_preference": (
                            preferred.candidate_id
                            if preferred.authenticity_review_band != other.authenticity_review_band and preferred.authenticity_review_band
                            else other.candidate_id
                            if preferred.authenticity_review_band != other.authenticity_review_band and other.authenticity_review_band
                            else ""
                        ),
                        "support_needed_preference": (
                            preferred.candidate_id
                            if preferred.support_needed_band != other.support_needed_band and preferred.support_needed_band
                            else other.candidate_id
                            if preferred.support_needed_band != other.support_needed_band and other.support_needed_band
                            else ""
                        ),
                    }
                )
    return pair_count


def write_summary(candidates: list[CandidateLabel], batches: list[dict[str, object]], pair_count: int, reviewed_at: str) -> None:
    recommendation_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for candidate in candidates:
        recommendation_counts[candidate.recommendation] = recommendation_counts.get(candidate.recommendation, 0) + 1
        source_counts[candidate.source_group] = source_counts.get(candidate.source_group, 0) + 1

    with SUMMARY_MD.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Bootstrap Label Artifacts Summary\n\n")
        handle.write(f"- Generated at: {reviewed_at}\n")
        handle.write(f"- Individual rows used: {len(candidates)}\n")
        handle.write(f"- Adjudicated rows written: {len(candidates)}\n")
        handle.write(f"- Pairwise rows written: {pair_count}\n")
        handle.write(f"- Batch shortlist tasks written: {len(batches)}\n")
        handle.write("- Caveat: these are bootstrap weak-label artifacts derived from one reviewer stream, not true multi-review committee ground truth.\n\n")
        handle.write("## Source Counts\n")
        for key in sorted(source_counts):
            handle.write(f"- {key}: {source_counts[key]}\n")
        handle.write("\n## Recommendation Counts\n")
        for key in sorted(recommendation_counts):
            handle.write(f"- {key}: {recommendation_counts[key]}\n")
        handle.write("\n## Batch Shape\n")
        handle.write(f"- batch_count: {len(batches)}\n")
        handle.write("- batch_size: 8\n")
        handle.write("- pairwise_per_batch: 28\n")
        handle.write("- task_type in pairwise_labels.csv: shortlist_first\n")
        handle.write(f"- targeted_batch_count: {sum(1 for batch in batches if str(batch['batch_id']).startswith('targeted_batch_'))}\n")


def main() -> None:
    reviewed_at = now_utc()
    candidates = load_candidates()
    write_adjudicated(candidates, reviewed_at)
    batches = build_batches(candidates, reviewed_at) + build_targeted_batches(candidates, reviewed_at)
    write_batches(batches)
    pair_count = write_pairwise(candidates, batches, reviewed_at)
    write_summary(candidates, batches, pair_count, reviewed_at)
    print(
        json.dumps(
            {
                "adjudicated_rows": len(candidates),
                "batch_tasks": len(batches),
                "pairwise_rows": pair_count,
                "summary_file": str(SUMMARY_MD),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
