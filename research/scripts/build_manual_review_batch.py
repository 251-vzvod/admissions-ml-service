"""Build a high-value manual review batch from draft annotations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _severity_score(reasons: list[str]) -> int:
    weights = {
        "needs_edit": 3,
        "authenticity_review": 2,
        "hidden_potential_high_priority": 2,
        "low_annotation_confidence": 2,
    }
    return sum(weights.get(reason, 1) for reason in reasons)


def _review_reasons(annotation: dict[str, Any]) -> list[str]:
    reasons: list[str] = []
    if annotation.get("triage_status") == "needs_edit":
        reasons.append("needs_edit")
    if bool(annotation.get("authenticity_review_flag", False)):
        reasons.append("authenticity_review")
    if bool(annotation.get("hidden_potential_flag", False)) and int(annotation.get("committee_priority", 0) or 0) >= 4:
        reasons.append("hidden_potential_high_priority")
    if str(annotation.get("confidence", "")).strip().lower() == "low":
        reasons.append("low_annotation_confidence")
    return reasons


def build_manual_review_batch(
    candidates_payload: dict[str, Any],
    annotations_payload: dict[str, Any],
) -> dict[str, Any]:
    candidates = candidates_payload.get("candidates", [])
    annotations = annotations_payload.get("annotations", [])
    if not isinstance(candidates, list) or not isinstance(annotations, list):
        raise ValueError("invalid input payloads")

    candidate_by_id = {
        str(item.get("candidate_id", "")).strip(): item for item in candidates if isinstance(item, dict)
    }

    selected_rows: list[dict[str, Any]] = []
    reason_counts: dict[str, int] = {
        "needs_edit": 0,
        "authenticity_review": 0,
        "hidden_potential_high_priority": 0,
        "low_annotation_confidence": 0,
    }

    for annotation in annotations:
        if not isinstance(annotation, dict):
            continue
        candidate_id = str(annotation.get("candidate_id", "")).strip()
        if not candidate_id or candidate_id not in candidate_by_id:
            continue

        reasons = _review_reasons(annotation)
        if not reasons:
            continue

        for reason in reasons:
            reason_counts[reason] += 1

        selected_rows.append(
            {
                "candidate_id": candidate_id,
                "review_reasons": reasons,
                "review_priority_score": _severity_score(reasons),
                "draft_annotation": annotation,
                "candidate_payload": candidate_by_id[candidate_id],
            }
        )

    selected_rows.sort(
        key=lambda item: (
            -int(item["review_priority_score"]),
            str(item["candidate_id"]),
        )
    )

    return {
        "meta": {
            "batch_version": "manual_review_batch_v1",
            "source_candidates_file": "data/archive/candidates.json",
            "source_annotations_file": "data/archive/annotation_pack_draft.json",
            "selection_policy": [
                "triage_status == needs_edit",
                "authenticity_review_flag == true",
                "hidden_potential_flag == true AND committee_priority >= 4",
                "confidence == low",
            ],
        },
        "summary": {
            "selected_candidate_count": len(selected_rows),
            "reason_counts": reason_counts,
        },
        "candidates": selected_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build manual review batch from draft annotations")
    parser.add_argument("--candidates", default="data/archive/candidates.json")
    parser.add_argument("--annotations", default="data/archive/annotation_pack_draft.json")
    parser.add_argument("--output", default="data/archive/manual_review_batch_v1.json")
    args = parser.parse_args()

    candidates_payload = _load_json(ROOT / args.candidates)
    annotations_payload = _load_json(ROOT / args.annotations)

    batch = build_manual_review_batch(candidates_payload, annotations_payload)
    output_path = ROOT / args.output
    output_path.write_text(json.dumps(batch, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Manual review batch generated: {output_path}")


if __name__ == "__main__":
    main()
